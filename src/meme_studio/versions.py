"""Immutable project snapshots with a transactional current-version pointer."""

import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

from .database import connect
from .errors import AppError
from .gallery import GalleryStore
from .image_files import png_bytes, verify_png
from .media_io import StoredFile, file_sha256, store_asset
from .schemas import Asset, AssetKind, Project, Region, Version, VersionKind, new_id, utc_now


def document(**fields) -> str:
    return json.dumps({"schema_version": 1, **fields}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _version(db, project_id: str, version_id: str) -> Version:
    row = db.execute("SELECT * FROM versions WHERE project_id = ? AND id = ?", (project_id, version_id)).fetchone()
    if row is None:
        raise AppError("VERSION_NOT_FOUND", "该版本不属于当前项目，或版本已不可用。")
    return Version(**row)


def _project(db, project_id: str) -> Project:
    row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        raise AppError("PROJECT_NOT_FOUND", "请先从图库创建或选择一个项目。")
    return Project(**row)


def _insert_version(db, snapshot: Version) -> None:
    db.execute(
        """INSERT INTO versions (id, project_id, parent_id, kind, base_image_asset_id, base_video_asset_id,
            anchor_image_version_id, text_layers_json, constraints_json, operation_json, job_id, created_at, region_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            snapshot.id, snapshot.project_id, snapshot.parent_id, snapshot.kind, snapshot.base_image_asset_id,
            snapshot.base_video_asset_id, snapshot.anchor_image_version_id, snapshot.text_layers_json,
            snapshot.constraints_json, snapshot.operation_json, snapshot.job_id, snapshot.created_at, snapshot.region_id,
        ),
    )


class ProjectStore:
    def __init__(self, gallery: GalleryStore):
        self.gallery = gallery
        self.runtime = gallery.runtime
        self.database = gallery.database

    def projects(self) -> list[Project]:
        with connect(self.database) as db:
            return [Project(**row) for row in db.execute("SELECT * FROM projects ORDER BY updated_at DESC, rowid DESC")]

    def create(self, gallery_id: str, image_id: str, title: str = "") -> Version:
        photo = self.gallery.get_photo(gallery_id, image_id)
        self.gallery.image_file(photo)
        title = title.strip() or photo.name[:160]
        if len(title) > 160:
            raise AppError("PROJECT_TITLE", "项目名称最多 160 个字符。")
        now = utc_now()
        project_id = new_id()
        root = Version(new_id(), project_id, VersionKind.ROOT, photo.asset_id, now)
        with connect(self.database) as db:
            db.execute("BEGIN IMMEDIATE")
            image = db.execute(
                """SELECT asset_id FROM gallery_images WHERE id = ? AND gallery_id = ?
                   AND removed_at IS NULL AND import_status = 'ready'""", (image_id, gallery_id),
            ).fetchone()
            if image is None or image["asset_id"] != photo.asset_id:
                raise AppError("PHOTO_NOT_FOUND", "原图库图片已变化，请重新选图。")
            db.execute(
                "INSERT INTO projects (id, title, source_image_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (project_id, title, photo.id, now, now),
            )
            _insert_version(db, root)
            db.execute(
                "UPDATE projects SET root_version_id = ?, current_version_id = ? WHERE id = ?",
                (root.id, root.id, project_id),
            )
        return root

    def current(self, project_id: str) -> Version:
        with connect(self.database) as db:
            db.execute("BEGIN")
            project = _project(db, project_id)
            return _version(db, project_id, project.current_version_id)

    def history(self, project_id: str) -> list[Version]:
        with connect(self.database) as db:
            return [
                Version(**row) for row in db.execute(
                    "SELECT * FROM versions WHERE project_id = ? ORDER BY rowid DESC", (project_id,),
                )
            ]

    def version(self, project_id: str, version_id: str) -> Version:
        with connect(self.database) as db:
            return _version(db, project_id, version_id)

    def require_current(self, project_id: str, expected_id: str) -> Version:
        current = self.current(project_id)
        if current.id != expected_id:
            raise AppError("STALE_VERSION", "当前版本已在另一个窗口改变，请重新打开项目。")
        return current

    def commit(
        self, parent: Version, kind: VersionKind, *, text_json: str | None = None,
        region: Region | None = None, clear_region: bool = False, base_image_asset_id: str | None = None,
        constraints_json: str | None = None,
    ) -> Version:
        base = base_image_asset_id or parent.base_image_asset_id
        if region is not None and (
            region.project_id != parent.project_id or region.source_version_id != parent.id
            or region.base_asset_id != base or base != parent.base_image_asset_id
        ):
            raise AppError("STALE_REGION", "选区与当前底图或版本不匹配。")
        keeps_region = base == parent.base_image_asset_id and not clear_region and kind != VersionKind.IMAGE_EDIT
        region_id = parent.region_id if keeps_region else None
        if region is not None:
            region_id = region.id
        child = replace(
            parent, id=new_id(), parent_id=parent.id, kind=kind, created_at=utc_now(),
            base_image_asset_id=base, text_layers_json=text_json if text_json is not None else parent.text_layers_json,
            constraints_json=parent.constraints_json if constraints_json is None else constraints_json,
            region_id=region_id, operation_json=document(action=str(kind)), job_id=None,
        )
        with connect(self.database) as db:
            db.execute("BEGIN IMMEDIATE")
            project = _project(db, parent.project_id)
            if project.current_version_id != parent.id:
                raise AppError("STALE_VERSION", "当前版本已改变，请重新打开项目后编辑。")
            if region is not None:
                db.execute(
                    """INSERT INTO regions (id, project_id, source_version_id, base_asset_id, base_asset_hash,
                       width, height, mask_asset_id, prompts_json, transform_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        region.id, region.project_id, region.source_version_id, region.base_asset_id,
                        region.base_asset_hash, region.width, region.height, region.mask_asset_id,
                        region.prompts_json, region.transform_json, region.created_at,
                    ),
                )
            _insert_version(db, child)
            db.execute(
                "UPDATE projects SET current_version_id = ?, updated_at = ?, revision = revision + 1 WHERE id = ?",
                (child.id, child.created_at, project.id),
            )
        return child

    def record_generated(
        self, db, parent: Version, asset_id: str, job_id: str, source_revision: int, frozen: dict,
    ) -> None:
        if db.execute("SELECT 1 FROM versions WHERE job_id = ?", (job_id,)).fetchone() is not None:
            return
        anchor = frozen.get("anchor_image_version_id")
        if anchor:
            result = replace(
                parent, id=new_id(), parent_id=parent.id, kind=VersionKind.VIDEO, created_at=utc_now(),
                base_video_asset_id=asset_id, anchor_image_version_id=anchor, region_id=None, job_id=job_id,
                constraints_json=frozen["result_constraints_json"],
                operation_json=document(action="video", origin="generated", prompt=frozen["prompt"],
                                        instruction=frozen["instruction"], parameters=frozen["parameters"]),
            )
        else:
            result = replace(
                parent, id=new_id(), parent_id=parent.id, kind=VersionKind.IMAGE_EDIT, created_at=utc_now(),
                base_image_asset_id=asset_id, base_video_asset_id=None, anchor_image_version_id=None,
                constraints_json=frozen.get("result_constraints_json", parent.constraints_json),
                region_id=None, operation_json=document(action="image_edit", function=frozen["function"],
                                                       prompt=frozen["prompt"]), job_id=job_id,
            )
        _insert_version(db, result)
        db.execute(
            """UPDATE projects SET current_version_id = ?, updated_at = ?, revision = revision + 1
               WHERE id = ? AND current_version_id = ? AND revision = ?""",
            (result.id, result.created_at, parent.project_id, parent.id, source_revision),
        )

    def _move(self, project_id: str, expected_id: str, target_id: str | None) -> Version:
        with connect(self.database) as db:
            db.execute("BEGIN IMMEDIATE")
            project = _project(db, project_id)
            current = _version(db, project_id, project.current_version_id)
            if current.id != expected_id:
                raise AppError("STALE_VERSION", "当前版本已改变，请重新打开项目。")
            destination = target_id if target_id is not None else current.parent_id
            if destination is None or destination == current.id:
                return current
            restored = _version(db, project_id, destination)
            db.execute(
                "UPDATE projects SET current_version_id = ?, updated_at = ?, revision = revision + 1 WHERE id = ?",
                (restored.id, utc_now(), project_id),
            )
            return restored

    def checkout(self, project_id: str, expected_id: str, target_id: str) -> Version:
        if not target_id:
            raise AppError("VERSION_NOT_FOUND", "请先选择要恢复的历史版本。")
        return self._move(project_id, expected_id, target_id)

    def undo(self, project_id: str, expected_id: str) -> Version:
        return self._move(project_id, expected_id, None)

    def asset(self, asset_id: str) -> Asset:
        with connect(self.database) as db:
            row = db.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            raise AppError("ASSET_NOT_FOUND", "项目的图像资产不存在。")
        return Asset(**row)

    def image_path(self, asset: Asset) -> Path:
        folder = "exports" if asset.kind == AssetKind.EXPORT else "assets"
        path = (self.runtime / asset.relative_path).resolve()
        if path != self.runtime / folder / f"{asset.sha256}.png" or not path.is_file():
            raise AppError("ASSET_NOT_FOUND", "项目的图像文件缺失或路径无效。")
        return path

    def load_image(self, asset: Asset) -> Image.Image:
        try:
            path = self.image_path(asset)
            if file_sha256(path) != asset.sha256:
                raise ValueError("hash mismatch")
            with Image.open(path, formats=["PNG"]) as image:
                if image.size != (asset.width, asset.height):
                    raise ValueError("size mismatch")
                return image.copy()
        except (OSError, ValueError) as exc:
            raise AppError("ASSET_DECODE", "项目的图像文件损坏或尺寸不符。") from exc

    def put_image(self, image: Image.Image, kind: AssetKind) -> Asset:
        folder = "exports" if kind == AssetKind.EXPORT else "assets"
        saved = store_asset(self.runtime, png_bytes(image), ".png", verify_png, folder=folder)
        return self.register_asset(saved, kind, "image/png", *image.size)

    def register_asset(self, saved: StoredFile, kind: AssetKind, mime: str, width: int, height: int,
                       duration: float | None = None) -> Asset:
        with connect(self.database) as db:
            db.execute(
                """INSERT INTO assets (id, sha256, relative_path, mime, width, height, duration, byte_size, kind, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(relative_path) DO NOTHING""",
                (new_id(), saved.sha256, saved.relative_path, mime, width, height, duration,
                 saved.byte_size, kind, utc_now()),
            )
            row = db.execute("SELECT * FROM assets WHERE relative_path = ?", (saved.relative_path,)).fetchone()
        return Asset(**row)

    def region(self, snapshot: Version) -> Region | None:
        if snapshot.region_id is None:
            return None
        with connect(self.database) as db:
            row = db.execute("SELECT * FROM regions WHERE id = ?", (snapshot.region_id,)).fetchone()
        asset = self.asset(snapshot.base_image_asset_id)
        if (
            row is None or row["project_id"] != snapshot.project_id or row["base_asset_id"] != asset.id
            or row["base_asset_hash"] != asset.sha256 or (row["width"], row["height"]) != (asset.width, asset.height)
        ):
            raise AppError("STALE_REGION", "历史选区与当前底图不匹配，请清空后重新选区。")
        return Region(**row)
