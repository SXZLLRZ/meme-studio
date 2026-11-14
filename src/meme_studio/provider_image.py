"""Static-provider image preparation and inverse mapping."""

from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from .errors import AppError
from .image_files import on_white, png_bytes


MIN_EDGE = 512
MAX_INPUT_EDGE = 1536
MAX_INPUT_BYTES = 10_000_000
MAX_RESULT_BYTES = 24_000_000
MAX_RESULT_PIXELS = 24_000_000
PREP_VERSION = "p4-rgb-white-pad-v1"


@dataclass(frozen=True)
class ImageGeometry:
    source: tuple[int, int]
    resized: tuple[int, int]
    sent: tuple[int, int]
    offset: tuple[int, int]


def prepare_image(base: Image.Image, mask: Image.Image | None) -> tuple[bytes, bytes | None, ImageGeometry]:
    if mask is not None:
        validate_mask(mask, base.size)
    low, high = min(base.size), max(base.size)
    scale = min(1.0, MAX_INPUT_EDGE / high)
    if low < MIN_EDGE and high * MIN_EDGE / low <= MAX_INPUT_EDGE:
        scale = MIN_EDGE / low
    resized = tuple(max(1, round(edge * scale)) for edge in base.size)
    sent = tuple(max(MIN_EDGE, edge) for edge in resized)
    offset = tuple((edge - fitted) // 2 for edge, fitted in zip(sent, resized))
    geometry = ImageGeometry(base.size, resized, sent, offset)

    fitted_base = on_white(base)
    if resized != base.size:
        fitted_base = fitted_base.resize(resized, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", sent, "white")
    canvas.paste(fitted_base, offset)
    image_data = png_bytes(canvas)
    mask_data = None
    if mask is not None:
        canvas_mask = Image.new("L", sent)
        canvas_mask.paste(mask.resize(resized, Image.Resampling.NEAREST), offset)
        validate_mask(canvas_mask, sent)
        mask_data = png_bytes(canvas_mask)
    if len(image_data) > MAX_INPUT_BYTES or (mask_data is not None and len(mask_data) > MAX_INPUT_BYTES):
        raise AppError("PROVIDER_INPUT_SIZE", "发送副本超过模型的 10 MB 限制。")
    return image_data, mask_data, geometry


def validate_mask(mask: Image.Image, size: tuple[int, int]) -> None:
    if mask.mode != "L" or mask.size != size:
        raise AppError("PROVIDER_MASK", "蒙版格式或尺寸与无字底图不匹配。")
    counts = mask.histogram()
    if counts[0] + counts[255] != size[0] * size[1] or counts[255] == 0:
        raise AppError("PROVIDER_MASK", "局部蒙版必须是非空的黑白二值图。")


def decode_result(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_RESULT_BYTES:
        raise AppError("RESULT_SIZE", "结果文件为空或超过本地下载上限。", retryable=True)
    try:
        with Image.open(BytesIO(data), formats=["PNG", "JPEG", "WEBP"]) as source:
            if source.width * source.height > MAX_RESULT_PIXELS or getattr(source, "n_frames", 1) != 1:
                raise AppError("RESULT_SIZE", "结果图片的尺寸或帧数超过限制。")
            source.verify()
        with Image.open(BytesIO(data), formats=["PNG", "JPEG", "WEBP"]) as source:
            result = source.convert("RGBA" if "A" in source.getbands() or "transparency" in source.info else "RGB")
            result.info.clear()
            return result
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise AppError("RESULT_DECODE", "下载的结果无法解码为图片。", retryable=True) from exc


def map_result(result: Image.Image, geometry: ImageGeometry, base: Image.Image, protect: Image.Image | None) -> Image.Image:
    if result.size != tuple(geometry.sent):
        if protect is not None:
            raise AppError("RESULT_GEOMETRY", "返回尺寸与发送副本不同，无法可靠对齐并保留选区外像素。")
        return result
    x, y = geometry.offset
    width, height = geometry.resized
    mapped = result.crop((x, y, x + width, y + height))
    if mapped.size != base.size:
        mapped = mapped.resize(base.size, Image.Resampling.LANCZOS)
    if protect is None:
        return mapped
    validate_mask(protect, base.size)
    return Image.composite(mapped.convert(base.mode), base, protect)
