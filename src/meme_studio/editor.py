"""Local editing, preview and PNG export over versioned clean assets."""

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageChops

from .constraints import constraint_document, constraint_summary, read_constraints, remaining_after
from .database import connect
from .errors import AppError
from .image_files import on_white
from .media_io import file_sha256
from .regions import (
    MASK_ALPHA_THRESHOLD, ViewTransform, box_mask, brush_layer, paint_mask, tint_mask, view_image,
)
from .schemas import Asset, AssetKind, Region, Version, VersionKind, new_id, utc_now
from .text_layers import RENDERER_VERSION, TextLayer, TextRenderer, text_document
from .versions import ProjectStore, document


@dataclass(frozen=True)
class EditView:
    snapshot: Version
    text: TextLayer
    preview: Image.Image
    region_preview: Image.Image
    brush_value: dict
    region_summary: str
    constraint_summary: str


@dataclass(frozen=True)
class PngExport:
    path: Path
    cached: bool


def _layers(snapshot: Version) -> tuple[TextLayer, ...]:
    return tuple(TextLayer(**layer) for layer in json.loads(snapshot.text_layers_json)["layers"])


class LocalEditor:
    def __init__(self, projects: ProjectStore, renderer: TextRenderer):
        self.projects = projects
        self.renderer = renderer

    def _base(self, snapshot: Version) -> tuple[Asset, Image.Image]:
        asset = self.projects.asset(snapshot.base_image_asset_id)
        return asset, self.projects.load_image(asset)

    def view(self, snapshot: Version) -> EditView:
        _, base = self._base(snapshot)
        transform = ViewTransform.for_base(base.size)
        layers = _layers(snapshot)
        preview_layers = self._static_layers(snapshot)
        preview = view_image(self.renderer.compose(base, preview_layers), transform)
        clean = view_image(base, transform)
        background = on_white(clean)
        region = self.projects.region(snapshot)
        strokes = []
        region_summary = "整图 · 当前无局部蒙版。"
        if region is not None:
            mask_asset = self.projects.asset(region.mask_asset_id)
            mask = self.projects.load_image(mask_asset)
            counts = mask.histogram()
            if (
                mask.mode != "L" or mask.size != base.size or counts[255] == 0
                or counts[0] + counts[255] != base.width * base.height
            ):
                raise AppError("REGION_MASK", "历史蒙版损坏，请清空选区后重新绘制。")
            clean = tint_mask(clean, mask)
            strokes = [brush_layer(transform, mask)]
            coverage = counts[255] / (base.width * base.height)
            region_summary = f"局部选区 · {counts[255]:,} 像素，占底图 {coverage:.1%}。"
        return EditView(
            snapshot, layers[0] if layers else TextLayer(), preview, clean,
            {"background": background, "layers": strokes, "composite": None},
            region_summary, constraint_summary(snapshot),
        )

    def _static_layers(self, snapshot: Version) -> tuple[TextLayer, ...]:
        if snapshot.anchor_image_version_id:
            snapshot = self.projects.version(snapshot.project_id, snapshot.anchor_image_version_id)
        return _layers(snapshot)

    def clean_view(self, snapshot: Version) -> Image.Image:
        _, base = self._base(snapshot)
        return on_white(view_image(base, ViewTransform.for_base(base.size)))

    def preview_text(self, snapshot: Version, style: TextLayer) -> Image.Image:
        _, base = self._base(snapshot)
        composed = self.renderer.compose(base, (style,))
        return view_image(composed, ViewTransform.for_base(base.size))

    def save_text(self, snapshot: Version, style: TextLayer) -> Version:
        if snapshot.base_video_asset_id is not None:
            asset = self.projects.asset(snapshot.base_video_asset_id)
            size = (asset.width, asset.height)
        else:
            _, base = self._base(snapshot)
            size = base.size
        fonts = self.renderer.resolve_fonts((style,))
        if fonts:
            style = replace(style, font_path=str(fonts[0].path))
        layers = (style,) if style.text.strip() else ()
        if layers:
            self.renderer.render_overlay(size, layers, fonts)
        if layers == _layers(snapshot):
            return self.projects.require_current(snapshot.project_id, snapshot.id)
        entries = read_constraints(snapshot)
        remaining = remaining_after(entries, "caption")
        return self.projects.commit(
            snapshot, VersionKind.TEXT_EDIT, text_json=text_document(layers),
            constraints_json=constraint_document(remaining) if remaining != entries else None,
        )

    def validate_point(self, snapshot: Version, coordinates: tuple[int, int]) -> tuple[int, int]:
        asset = self.projects.asset(snapshot.base_image_asset_id)
        return ViewTransform.for_base((asset.width, asset.height)).point(coordinates)

    def save_box(self, snapshot: Version, first: tuple[int, int], second: tuple[int, int]) -> Version:
        asset, base = self._base(snapshot)
        transform = ViewTransform.for_base(base.size)
        mask, prompts = box_mask(transform, first, second)
        return self._save_region(snapshot, asset, transform, mask, prompts)

    def save_paint(self, snapshot: Version, value: dict | None) -> Version:
        asset, base = self._base(snapshot)
        transform = ViewTransform.for_base(base.size)
        expected = on_white(view_image(base, transform))
        background = value.get("background") if value else None
        if (
            background is None or background.size != transform.view_size
            or ImageChops.difference(background.convert("RGB"), expected).getbbox()
        ):
            raise AppError("STALE_REGION", "画布底图与当前版本不一致，请重新打开项目后涂抹。")
        mask = paint_mask(transform, value.get("layers") or [])
        prompts = {"mode": "paint", "alpha_threshold": MASK_ALPHA_THRESHOLD}
        return self._save_region(snapshot, asset, transform, mask, prompts)

    def _save_region(self, snapshot, asset, transform, mask, prompts) -> Version:
        self.projects.require_current(snapshot.project_id, snapshot.id)
        stored = self.projects.put_image(mask, AssetKind.MASK)
        region = Region(
            new_id(), snapshot.project_id, snapshot.id, asset.id, asset.sha256, *mask.size,
            stored.id, document(**prompts), document(**transform.metadata()), utc_now(),
        )
        return self.projects.commit(snapshot, VersionKind.REGION_EDIT, region=region)

    def clear_region(self, snapshot: Version) -> Version:
        if snapshot.region_id is None:
            return self.projects.require_current(snapshot.project_id, snapshot.id)
        return self.projects.commit(snapshot, VersionKind.REGION_EDIT, clear_region=True)

    def export_png(self, snapshot: Version) -> PngExport:
        self.projects.require_current(snapshot.project_id, snapshot.id)
        asset = self.projects.asset(snapshot.base_image_asset_id)
        layers = self._static_layers(snapshot)
        fonts = self.renderer.resolve_fonts(layers)
        key = document(
            base_sha256=asset.sha256, size=[asset.width, asset.height], text=text_document(layers),
            fonts=[font.digest for font in fonts], renderer=RENDERER_VERSION,
        )
        cache_options = document(cache_key=sha256(key.encode("utf-8")).hexdigest())
        with connect(self.projects.database) as db:
            cached = db.execute(
                "SELECT asset_id FROM exports WHERE version_id = ? AND format = 'png' AND options_json = ?",
                (snapshot.id, cache_options),
            ).fetchone()
        if cached is not None:
            try:
                output = self.projects.asset(cached["asset_id"])
                path = self.projects.image_path(output)
                if file_sha256(path) == output.sha256:
                    return PngExport(path, True)
            except (AppError, OSError):
                pass

        result = self.renderer.compose(self.projects.load_image(asset), layers, fonts)
        output = self.projects.put_image(result, AssetKind.EXPORT)
        with connect(self.projects.database) as db:
            db.execute(
                """INSERT INTO exports (id, version_id, format, options_json, asset_id, created_at)
                   VALUES (?, ?, 'png', ?, ?, ?)
                   ON CONFLICT(version_id, format, options_json) DO UPDATE SET asset_id = excluded.asset_id""",
                (new_id(), snapshot.id, cache_options, output.id, utc_now()),
            )
        return PngExport(self.projects.image_path(output), False)
