"""Render independent, measured text layers with system fonts."""

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from fontTools.ttLib import TTFont, TTLibError
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .errors import AppError
from .media_io import file_sha256
from .versions import document


RENDERER_VERSION = "p3-pillow-v1"


@dataclass(frozen=True)
class TextLayer:
    text: str = ""
    size_ratio: float = 0.12
    x: float = 0.5
    y: float = 1.0
    fill: str = "#FFFFFF"
    stroke_fill: str = "#000000"
    stroke_ratio: float = 0.006
    font_path: str | None = None

    def __post_init__(self) -> None:
        if len(self.text) > 600:
            raise AppError("TEXT_LIMIT", "文案最多 600 个字符。")
        if not (0.03 <= self.size_ratio <= 0.40 and 0 <= self.x <= 1 and 0 <= self.y <= 1
                and 0 <= self.stroke_ratio <= 0.03):
            raise AppError("TEXT_STYLE", "字号、位置或描边参数超出范围。")
        try:
            ImageColor.getrgb(self.fill)
            ImageColor.getrgb(self.stroke_fill)
        except (TypeError, ValueError) as exc:
            raise AppError("TEXT_COLOR", "请使用有效的文字和描边颜色。") from exc


def text_document(layers: Iterable[TextLayer]) -> str:
    return document(layers=[asdict(layer) for layer in layers])


@dataclass(frozen=True)
class FontInfo:
    path: Path
    characters: frozenset[int]
    digest: str


@lru_cache(maxsize=12)
def _font_info(path: Path, modified: int, size: int) -> FontInfo:
    with TTFont(str(path), fontNumber=0, lazy=True) as font:
        return FontInfo(path, frozenset(font.getBestCmap() or {}), file_sha256(path))


class TextRenderer:
    def __init__(self, configured_font: Path | None = None):
        if configured_font:
            self.font_paths = (configured_font,)
        else:
            directory = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
            self.font_paths = tuple(directory / name for name in ("msyh.ttc", "simhei.ttf", "Deng.ttf", "simsun.ttc"))

    def resolve_font(self, layer: TextLayer) -> FontInfo:
        needed = {ord(character) for character in layer.text if character not in "\r\n"}
        supported: set[int] = set()
        found = False
        candidates = (Path(layer.font_path),) if layer.font_path else self.font_paths
        for path in candidates:
            try:
                stat = path.stat()
                info = _font_info(path, stat.st_mtime_ns, stat.st_size)
            except (OSError, ValueError, TTLibError):
                continue
            found = True
            if needed.issubset(info.characters):
                return info
            supported.update(info.characters)
        if not found:
            raise AppError("FONT_MISSING", "找不到可用字体，请在设置中配置本机中文字体路径。")
        examples = "".join(chr(codepoint) for codepoint in sorted(needed - supported)[:8])
        message = f"当前字体缺少字符「{examples}」。" if examples else "文案中的字符无法用同一字体显示。"
        raise AppError("FONT_GLYPH", message + "请换字体或替换这些字符。")

    def resolve_fonts(self, layers: Iterable[TextLayer]) -> tuple[FontInfo, ...]:
        return tuple(self.resolve_font(layer) for layer in layers if layer.text.strip())

    def render_overlay(
        self, size: tuple[int, int], layers: tuple[TextLayer, ...], fonts: tuple[FontInfo, ...] | None = None,
    ) -> Image.Image:
        overlay = Image.new("RGBA", size)
        visible = (layer for layer in layers if layer.text.strip())
        fonts = self.resolve_fonts(layers) if fonts is None else fonts
        for layer, info in zip(visible, fonts, strict=True):
            _draw_layer(overlay, layer, info)
        return overlay

    def compose(
        self, base: Image.Image, layers: tuple[TextLayer, ...], fonts: tuple[FontInfo, ...] | None = None,
    ) -> Image.Image:
        if not any(layer.text.strip() for layer in layers):
            return base.copy()
        overlay = self.render_overlay(base.size, layers, fonts)
        result = Image.alpha_composite(base.convert("RGBA"), overlay)
        return result if base.mode == "RGBA" else result.convert("RGB")


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, stroke: int, width: int) -> str | None:
    output = []
    for paragraph in text.split("\n"):
        current = ""
        for character in paragraph:
            attempt = current + character
            left, _, right, _ = draw.textbbox((0, 0), attempt, font=font, stroke_width=stroke)
            if right - left <= width:
                current = attempt
            elif current:
                output.append(current)
                current = character
            else:
                return None
        output.append(current)
    return "\n".join(output)


def _draw_layer(overlay: Image.Image, layer: TextLayer, info: FontInfo) -> None:
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    short_edge = min(overlay.size)
    margin = max(1, round(short_edge * 0.05))
    available_width, available_height = width - 2 * margin, height - 2 * margin
    requested = max(1, round(short_edge * layer.size_ratio))
    text = layer.text.replace("\r\n", "\n").replace("\r", "\n")
    # Try the requested size first and shrink until both wrapped width and height fit.
    size = requested
    minimum = min(requested, 8)
    try:
        while True:
            font = ImageFont.truetype(str(info.path), size=size)
            stroke = min(round(short_edge * layer.stroke_ratio), max(0, size // 4))
            spacing = max(1, round(size * 0.18))
            wrapped = _wrap(draw, text, font, stroke, available_width)
            if wrapped is not None:
                left, top, right, bottom = draw.multiline_textbbox(
                    (0, 0), wrapped, font=font, stroke_width=stroke, spacing=spacing, align="center",
                )
                actual_width, actual_height = right - left, bottom - top
                if actual_width <= available_width and actual_height <= available_height:
                    x = margin + round((available_width - actual_width) * layer.x) - left
                    y = margin + round((available_height - actual_height) * layer.y) - top
                    draw.multiline_text(
                        (x, y), wrapped, font=font, fill=layer.fill, stroke_width=stroke,
                        stroke_fill=layer.stroke_fill, spacing=spacing, align="center",
                    )
                    return
            if size <= minimum:
                break
            size = max(minimum, int(size * 0.84))
    except OSError as exc:
        raise AppError("FONT_LOAD", "字体无法渲染，请检查字体文件。") from exc
    raise AppError("TEXT_FIT", "文案无法放进图片的安全范围，请缩短文案或换一张尺寸较大的图片。")
