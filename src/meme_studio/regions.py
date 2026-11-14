"""Map local editor pixels to an immutable base image and create binary masks."""

from dataclasses import dataclass
from math import ceil, floor

from PIL import Image, ImageChops, ImageDraw

from .errors import AppError


EDITOR_EDGE = 1024
MASK_ALPHA_THRESHOLD = 128


@dataclass(frozen=True)
class ViewTransform:
    base_size: tuple[int, int]
    view_size: tuple[int, int]

    @classmethod
    def for_base(cls, size: tuple[int, int]) -> "ViewTransform":
        factor = min(1.0, EDITOR_EDGE / max(size))
        return cls(size, tuple(max(1, round(edge * factor)) for edge in size))

    def point(self, point: tuple[int, int]) -> tuple[int, int]:
        if len(point) != 2 or any(type(p) is not int or not 0 <= p < edge for p, edge in zip(point, self.view_size)):
            raise AppError("REGION_POINT", "请点击图片以内的位置，避开留白。")
        return point

    def box(self, first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int, int, int]:
        x0, y0 = self.point(first)
        x1, y1 = self.point(second)
        if x0 == x1 or y0 == y1:
            raise AppError("REGION_BOX", "两个角需要围出有宽度和高度的矩形。")
        sx, sy = (base / view for base, view in zip(self.base_size, self.view_size))
        return (floor(min(x0, x1) * sx), floor(min(y0, y1) * sy),
                ceil((max(x0, x1) + 1) * sx), ceil((max(y0, y1) + 1) * sy))

    def metadata(self) -> dict:
        return {
            "coordinate_space": "preview_image_pixels",
            "base_size": self.base_size,
            "view_size": self.view_size,
            "view_to_base_scale": [base / view for base, view in zip(self.base_size, self.view_size)],
            "offset": [0, 0],
        }


def box_mask(transform: ViewTransform, first: tuple[int, int], second: tuple[int, int]) -> tuple[Image.Image, dict]:
    x0, y0, x1, y1 = transform.box(first, second)
    mask = Image.new("L", transform.base_size, 0)
    ImageDraw.Draw(mask).rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
    return mask, {"mode": "box", "view_points": [first, second], "base_box": [x0, y0, x1, y1]}


def paint_mask(transform: ViewTransform, layers: list[Image.Image]) -> Image.Image:
    alpha = Image.new("L", transform.view_size)
    for layer in layers:
        if layer.size != transform.view_size or "A" not in layer.getbands():
            raise AppError("REGION_LAYER", "涂抹画布尺寸或笔画图层已改变，请重新加载。")
        alpha = ImageChops.lighter(alpha, layer.getchannel("A"))
    binary = alpha.point([0] * MASK_ALPHA_THRESHOLD + [255] * (256 - MASK_ALPHA_THRESHOLD))
    mask = binary.resize(transform.base_size, Image.Resampling.NEAREST)
    if not mask.getbbox():
        raise AppError("REGION_EMPTY", "局部选区为空。请添加笔画，或选择整图/清空选区。")
    return mask


def view_image(image: Image.Image, transform: ViewTransform) -> Image.Image:
    if image.size == transform.view_size:
        return image.copy()
    return image.resize(transform.view_size, Image.Resampling.LANCZOS)


def tint_mask(preview: Image.Image, mask: Image.Image) -> Image.Image:
    opacity = mask.resize(preview.size, Image.Resampling.NEAREST).point([0] * 255 + [95])
    tint = Image.new("RGBA", preview.size, (255, 62, 46, 0))
    tint.putalpha(opacity)
    return Image.alpha_composite(preview.convert("RGBA"), tint)


def brush_layer(transform: ViewTransform, mask: Image.Image) -> Image.Image:
    layer = Image.new("RGBA", transform.view_size, (255, 62, 46, 0))
    layer.putalpha(mask.resize(transform.view_size, Image.Resampling.NEAREST))
    return layer
