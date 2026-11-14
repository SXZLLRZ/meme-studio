"""Incremental vector cache and validated, atomically published gallery indexes."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .database import connect
from .errors import AppError
from .gallery import GalleryInfo, GalleryStore, Photo
from .media_io import file_sha256
from .model_manager import BATCH_SIZE, ModelSignature, unit_rows
from .schemas import new_id


class Encoder(Protocol):
    def signature(self) -> ModelSignature: ...
    def encode_images(self, paths: list[Path]) -> np.ndarray: ...
    def encode_text(self, query: str) -> np.ndarray: ...


@dataclass(frozen=True)
class IndexResult:
    generation: str | None
    count: int
    encoded: int = 0
    dimension: int | None = None
    repaired: bool = False


@dataclass(frozen=True)
class SearchHit:
    photo: Photo
    score: float


def _rows(photos: tuple[Photo, ...]) -> list[list[str]]:
    return [[photo.id, photo.asset_hash] for photo in photos]


def _manifest(generation: str, signature: ModelSignature, photos: tuple[Photo, ...], matrix_hash: str) -> dict:
    return {
        "schema_version": 1,
        "generation": generation,
        "model": list(signature.key()),
        "rows": _rows(photos),
        "matrix_sha256": matrix_hash,
    }


def _same_signature(active, signature: ModelSignature) -> bool:
    return tuple(active[field] for field in ("model_name", "checkpoint_hash", "preprocess_version")) == signature.key()


class GalleryRetrieval:
    def __init__(self, gallery: GalleryStore, encoder: Encoder):
        self.gallery = gallery
        self.encoder = encoder
        self.indexes = gallery.runtime / "indexes"

    def _active(self, gallery_id: str):
        with connect(self.gallery.database) as db:
            return db.execute(
                "SELECT * FROM feature_sets WHERE gallery_id = ? AND status = 'active'", (gallery_id,),
            ).fetchone()

    def _load(self, active, signature: ModelSignature, photos: tuple[Photo, ...]) -> np.ndarray:
        try:
            generation = active["generation"]
            index_path = (self.gallery.runtime / active["index_path"]).resolve()
            manifest_path = (self.gallery.runtime / active["manifest_path"]).resolve()
            expected_dir = self.indexes / generation
            if index_path != expected_dir / "matrix.npy" or manifest_path != expected_dir / "manifest.json":
                raise ValueError("index path")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest != _manifest(generation, signature, photos, file_sha256(index_path)):
                raise ValueError("manifest")
            with index_path.open("rb") as stream:
                matrix = np.load(stream, allow_pickle=False)
            if matrix.dtype != np.float32 or matrix.shape != (len(photos), active["dimension"]):
                raise ValueError("index shape")
            if not np.isfinite(matrix).all():
                raise ValueError("index contents")
            with connect(self.gallery.database) as db:
                mapped = db.execute(
                    """SELECT image_id, asset_hash FROM image_features
                       WHERE feature_set_id = ? ORDER BY row_index""", (active["id"],),
                ).fetchall()
            if [list(row) for row in mapped] != _rows(photos):
                raise ValueError("row mapping")
            return matrix
        except (OSError, ValueError, EOFError, AttributeError) as exc:
            raise AppError("INDEX_DAMAGED", "索引文件损坏，请点击「更新/重建索引」。", retryable=True) from exc

    def _current(self, info: GalleryInfo, photos: tuple[Photo, ...], signature: ModelSignature):
        active = self._active(info.id)
        if active is None or active["gallery_revision"] != info.revision or not _same_signature(active, signature):
            raise AppError("INDEX_STALE", "图库尚无可用索引，请点击「更新/重建索引」。", retryable=True)
        return active, self._load(active, signature, photos)

    def status(self, gallery_id: str) -> IndexResult:
        info, photos = self.gallery.catalog(gallery_id)
        if not photos:
            return IndexResult(None, 0)
        signature = self.encoder.signature()
        active, matrix = self._current(info, photos, signature)
        return IndexResult(active["generation"], len(photos), dimension=matrix.shape[1])

    def _cached(self, signature: ModelSignature, photos: tuple[Photo, ...]) -> dict[str, np.ndarray]:
        vectors: dict[str, np.ndarray] = {}
        with connect(self.gallery.database) as db:
            for photo in photos:
                cached = db.execute(
                    """SELECT dimension, vector FROM image_vectors WHERE asset_hash = ?
                       AND model_name = ? AND checkpoint_hash = ? AND preprocess_version = ?""",
                    (photo.asset_hash, *signature.key()),
                ).fetchone()
                if cached is not None:
                    vector = np.frombuffer(cached["vector"], dtype="<f4")
                    if vector.size == cached["dimension"] and np.isclose(np.linalg.norm(vector), 1.0):
                        vectors[photo.asset_hash] = vector
        return vectors

    def ensure_index(self, gallery_id: str, *, force: bool = False) -> IndexResult:
        info, photos = self.gallery.catalog(gallery_id)
        if not photos:
            self._switch(info, (), None, None, None, None)
            return IndexResult(None, 0)

        signature = self.encoder.signature()
        repaired = False
        try:
            active, matrix = self._current(info, photos, signature)
            if not force:
                return IndexResult(active["generation"], len(photos), dimension=matrix.shape[1])
        except AppError as exc:
            if exc.code not in {"INDEX_STALE", "INDEX_DAMAGED"}:
                raise
            repaired = exc.code == "INDEX_DAMAGED"

        vectors = self._cached(signature, photos)
        missing = [photo for photo in photos if photo.asset_hash not in vectors]
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start:start + BATCH_SIZE]
            features = unit_rows(self.encoder.encode_images([self.gallery.image_file(photo) for photo in batch]))
            if features.shape[0] != len(batch):
                raise AppError("VECTOR_COUNT", "编码结果数量与照片数量不一致。")
            with connect(self.gallery.database) as db:
                for photo, vector in zip(batch, features, strict=True):
                    db.execute(
                        """INSERT INTO image_vectors VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(asset_hash, model_name, checkpoint_hash, preprocess_version)
                           DO UPDATE SET dimension = excluded.dimension, vector = excluded.vector""",
                        (photo.asset_hash, *signature.key(), vector.size, vector.astype("<f4").tobytes()),
                    )
                    vectors[photo.asset_hash] = vector

        try:
            matrix = np.stack([vectors[photo.asset_hash] for photo in photos]).astype(np.float32, copy=False)
        except ValueError as exc:
            raise AppError("VECTOR_SHAPE", "缓存向量维度不一致，原索引未切换。") from exc
        generation = self._publish(info, photos, signature, matrix)
        return IndexResult(generation, len(photos), len(missing), matrix.shape[1], repaired)

    def _publish(self, info: GalleryInfo, photos: tuple[Photo, ...], signature: ModelSignature, matrix: np.ndarray) -> str:
        generation = new_id()
        if self.indexes.resolve() != self.indexes:
            raise AppError("INDEX_PATH", "索引目录不能重定向到其他位置。")
        directory = self.indexes / generation
        matrix_file = directory / "matrix.npy"
        manifest_file = directory / "manifest.json"
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise AppError("INDEX_WRITE", "无法创建索引目录，原索引未切换。", retryable=True) from exc
        published = False
        try:
            with matrix_file.open("xb") as stream:
                np.save(stream, matrix, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            manifest = _manifest(generation, signature, photos, file_sha256(matrix_file))
            with manifest_file.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            self._switch(info, photos, signature, matrix.shape[1], matrix_file, manifest_file)
            published = True
        except OSError as exc:
            raise AppError("INDEX_WRITE", "无法保存新索引，原索引未切换。", retryable=True) from exc
        finally:
            if not published:
                matrix_file.unlink(missing_ok=True)
                manifest_file.unlink(missing_ok=True)
                directory.rmdir()
        return generation

    def _switch(self, info, photos, signature, dimension, index_file, manifest_file) -> None:
        with connect(self.gallery.database) as db:
            db.execute("BEGIN IMMEDIATE")
            revision = db.execute("SELECT revision FROM galleries WHERE id = ?", (info.id,)).fetchone()
            if revision is None or revision[0] != info.revision:
                raise AppError("INDEX_CONFLICT", "图库在建索引时有变化，请重试。", retryable=True)
            db.execute("UPDATE feature_sets SET status = 'retired' WHERE gallery_id = ? AND status = 'active'", (info.id,))
            if index_file is None:
                return
            index_id = new_id()
            db.execute(
                """INSERT INTO feature_sets
                   (id, gallery_id, model_name, checkpoint_hash, preprocess_version, dimension,
                    index_path, manifest_path, generation, status, gallery_revision)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
                (index_id, info.id, *signature.key(), dimension, index_file.relative_to(self.gallery.runtime).as_posix(),
                 manifest_file.relative_to(self.gallery.runtime).as_posix(), index_file.parent.name, info.revision),
            )
            db.executemany(
                "INSERT INTO image_features (image_id, feature_set_id, row_index, asset_hash) VALUES (?, ?, ?, ?)",
                [(photo.id, index_id, i, photo.asset_hash) for i, photo in enumerate(photos)],
            )

    def search(self, gallery_id: str, query: str, top_k: int, tags: tuple[str, ...] = ()) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise AppError("EMPTY_QUERY", "请输入中文描述，或用「浏览全部」手动选图。")
        if len(query) > 300 or not 1 <= top_k <= 50:
            raise AppError("QUERY_LIMIT", "描述最多 300 个字符，候选数为 1 至 50。")
        info, photos = self.gallery.catalog(gallery_id)
        if not photos:
            raise AppError("EMPTY_GALLERY", "当前图库没有图片，请先导入。")
        signature = self.encoder.signature()
        _, matrix = self._current(info, photos, signature)
        required = set(tags)
        rows = [index for index, photo in enumerate(photos) if required.issubset(photo.tags)]
        if not rows:
            return []
        query_vectors = unit_rows(self.encoder.encode_text(query))
        if query_vectors.shape != (1, matrix.shape[1]):
            raise AppError("QUERY_DIMENSION", "查询与索引的模型维度不匹配，请重建索引。")
        scores = np.clip(matrix[rows] @ query_vectors[0], -1.0, 1.0)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [SearchHit(photos[rows[index]], float(scores[index])) for index in order]
