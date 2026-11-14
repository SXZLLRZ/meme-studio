"""Manage gallery memberships without editing or deleting source photos."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .database import connect
from .errors import AppError
from .image_files import THUMBNAIL_SIZE, normalize_photo, png_bytes, read_source, verify_png
from .media_io import store_asset
from .schemas import new_id, utc_now


@dataclass(frozen=True)
class GalleryInfo:
    id: str
    name: str
    source_dir: str | None
    revision: int


@dataclass(frozen=True)
class Photo:
    id: str
    gallery_id: str
    name: str
    asset_id: str
    asset_hash: str
    image_path: str
    thumbnail_path: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class ImportResult:
    added: int
    duplicates: int
    failed: tuple[tuple[str, str], ...]


_PHOTOS = """
    SELECT gi.id, gi.gallery_id, gi.display_name, gi.thumbnail_path, gi.tags_json,
           a.id AS asset_id, a.sha256, a.relative_path
    FROM gallery_images AS gi JOIN assets AS a ON a.id = gi.asset_id
    WHERE gi.gallery_id = ? AND gi.import_status = 'ready'
"""


def _photo(row) -> Photo:
    tags = json.loads(row["tags_json"])["tags"]
    return Photo(
        row["id"], row["gallery_id"], row["display_name"], row["asset_id"], row["sha256"],
        row["relative_path"], row["thumbnail_path"], tuple(tags),
    )


class GalleryStore:
    def __init__(self, runtime: Path, max_bytes: int = 24 * 1024**2, max_pixels: int = 24_000_000):
        self.runtime = runtime.resolve()
        self.database = self.runtime / "app.sqlite3"
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels

    def galleries(self) -> list[GalleryInfo]:
        with connect(self.database) as db:
            return [
                GalleryInfo(**row) for row in db.execute(
                    "SELECT id, name, source_dir, revision FROM galleries ORDER BY created_at, id"
                )
            ]

    def create(self, name: str, source_dir: Path | None = None) -> GalleryInfo:
        name = name.strip()
        if not name or len(name) > 100:
            raise AppError("GALLERY_NAME", "请填写 1 至 100 个字符的图库名称。")
        gallery = GalleryInfo(new_id(), name, str(source_dir) if source_dir else None, 0)
        with connect(self.database) as db:
            db.execute(
                "INSERT INTO galleries (id, name, source_dir, created_at) VALUES (?, ?, ?, ?)",
                (gallery.id, gallery.name, gallery.source_dir, utc_now()),
            )
        return gallery

    def catalog(self, gallery_id: str) -> tuple[GalleryInfo, tuple[Photo, ...]]:
        with connect(self.database) as db:
            db.execute("BEGIN")
            gallery = db.execute(
                "SELECT id, name, source_dir, revision FROM galleries WHERE id = ?", (gallery_id,),
            ).fetchone()
            if gallery is None:
                raise AppError("GALLERY_NOT_FOUND", "图库不存在，请重新选择。")
            rows = db.execute(_PHOTOS + " AND gi.removed_at IS NULL ORDER BY gi.id", (gallery_id,)).fetchall()
        return GalleryInfo(**gallery), tuple(_photo(row) for row in rows)

    def get_photo(self, gallery_id: str, image_id: str, *, include_removed: bool = False) -> Photo:
        query = _PHOTOS + " AND gi.id = ?"
        if not include_removed:
            query += " AND gi.removed_at IS NULL"
        with connect(self.database) as db:
            row = db.execute(query, (gallery_id, image_id)).fetchone()
        if row is None:
            raise AppError("PHOTO_NOT_FOUND", "图片已移出或不属于当前图库，请重新选择。")
        return _photo(row)

    def image_file(self, photo: Photo, *, thumbnail: bool = False) -> Path:
        folder = "thumbnails" if thumbnail else "assets"
        relative = photo.thumbnail_path if thumbnail else photo.image_path
        path = (self.runtime / relative).resolve()
        if path.parent != self.runtime / folder or path.suffix != ".png" or not path.is_file():
            raise AppError("PHOTO_FILE", "项目中的图片副本缺失，请从原文件重新导入。")
        return path

    def import_directory(self, gallery_id: str, source: Path) -> ImportResult:
        try:
            source = source.expanduser().resolve()
            files = sorted(
                (file for file in source.iterdir() if file.is_file() and not file.is_symlink()
                 and not file.name.startswith(".")),
                key=lambda path: path.name.casefold(),
            )
        except (OSError, ValueError) as exc:
            raise AppError("GALLERY_DIRECTORY", "无法打开图库目录，请检查路径和权限。") from exc
        result = self.import_files(gallery_id, files)
        with connect(self.database) as db:
            db.execute("UPDATE galleries SET source_dir = ? WHERE id = ?", (str(source), gallery_id))
        return result

    def import_files(self, gallery_id: str, files: Iterable[Path]) -> ImportResult:
        with connect(self.database) as db:
            if not db.execute("SELECT 1 FROM galleries WHERE id = ?", (gallery_id,)).fetchone():
                raise AppError("GALLERY_NOT_FOUND", "图库不存在，请重新选择。")
        added = duplicates = 0
        failed: list[tuple[str, str]] = []
        for file in files:
            try:
                if self._import_one(gallery_id, file):
                    added += 1
                else:
                    duplicates += 1
            except AppError as exc:
                failed.append((file.name, exc.user_message))
        return ImportResult(added, duplicates, tuple(failed))

    def _import_one(self, gallery_id: str, file: Path) -> bool:
        source, digest = read_source(file, self.max_bytes)
        with connect(self.database) as db:
            prior = db.execute(
                _PHOTOS + " AND gi.source_hash = ? AND gi.removed_at IS NULL", (gallery_id, digest),
            ).fetchone()
        if prior is not None:
            try:
                photo = _photo(prior)
                self.image_file(photo)
                self.image_file(photo, thumbnail=True)
                return False
            except AppError:
                # Reimport repairs a missing managed copy from the same source.
                pass

        image = normalize_photo(source, self.max_pixels)
        original = store_asset(self.runtime, png_bytes(image), ".png", verify_png)
        preview = image.copy()
        preview.thumbnail(THUMBNAIL_SIZE)
        thumb = store_asset(self.runtime, png_bytes(preview), ".png", verify_png, folder="thumbnails")

        with connect(self.database) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO assets (id, sha256, relative_path, mime, width, height, byte_size, kind, created_at)
                   VALUES (?, ?, ?, 'image/png', ?, ?, ?, 'image', ?)
                   ON CONFLICT(relative_path) DO NOTHING""",
                (new_id(), original.sha256, original.relative_path, *image.size, original.byte_size, utc_now()),
            )
            asset_id = db.execute(
                "SELECT id FROM assets WHERE relative_path = ?", (original.relative_path,),
            ).fetchone()[0]
            existing = db.execute(
                """SELECT id, removed_at FROM gallery_images
                   WHERE gallery_id = ? AND (source_hash = ? OR asset_id = ?)""",
                (gallery_id, digest, asset_id),
            ).fetchone()
            if existing is not None:
                if existing["removed_at"] is None:
                    return False
                db.execute("UPDATE gallery_images SET removed_at = NULL WHERE id = ?", (existing["id"],))
            else:
                db.execute(
                    """INSERT INTO gallery_images
                       (id, gallery_id, source_path, source_hash, asset_id, tags_json, import_status,
                        created_at, display_name, thumbnail_path)
                       VALUES (?, ?, ?, ?, ?, '{"schema_version":1,"tags":[]}', 'ready', ?, ?, ?)""",
                    (new_id(), gallery_id, str(file), digest, asset_id, utc_now(), file.name, thumb.relative_path),
                )
            db.execute("UPDATE galleries SET revision = revision + 1 WHERE id = ?", (gallery_id,))
        return True

    def set_tags(self, gallery_id: str, image_id: str, values: Iterable[str]) -> Photo:
        tags = list(dict.fromkeys(tag.strip() for tag in values if tag.strip()))
        if len(tags) > 20 or any(len(tag) > 40 for tag in tags):
            raise AppError("PHOTO_TAGS", "每张图片最多 20 个标签，每个标签最多 40 个字符。")
        encoded = json.dumps({"schema_version": 1, "tags": tags}, ensure_ascii=False)
        with connect(self.database) as db:
            updated = db.execute(
                "UPDATE gallery_images SET tags_json = ? WHERE gallery_id = ? AND id = ? AND removed_at IS NULL",
                (encoded, gallery_id, image_id),
            ).rowcount
            if not updated:
                raise AppError("PHOTO_NOT_FOUND", "图片已移出或不属于当前图库，请重新选择。")
        return self.get_photo(gallery_id, image_id)

    def remove(self, gallery_id: str, image_id: str) -> None:
        with connect(self.database) as db:
            updated = db.execute(
                "UPDATE gallery_images SET removed_at = ? WHERE gallery_id = ? AND id = ? AND removed_at IS NULL",
                (utc_now(), gallery_id, image_id),
            ).rowcount
            if not updated:
                raise AppError("PHOTO_NOT_FOUND", "图片已移出或不属于当前图库，请重新选择。")
            db.execute("UPDATE galleries SET revision = revision + 1 WHERE id = ?", (gallery_id,))
