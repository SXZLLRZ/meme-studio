"""Decode user photos into metadata-free, orientation-corrected local PNGs."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

from .errors import AppError


EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}
THUMBNAIL_SIZE = (320, 320)


def read_source(path: Path, max_bytes: int) -> tuple[bytes, str]:
    if path.suffix.lower() not in EXTENSIONS:
        raise AppError("IMAGE_FORMAT", "支持 JPEG、PNG、WEBP 和 BMP。")
    try:
        with path.open("rb") as stream:
            data = stream.read(max_bytes + 1)
    except OSError as exc:
        raise AppError("IMAGE_READ", "文件无法读取。") from exc
    if len(data) > max_bytes:
        raise AppError("IMAGE_SIZE", f"文件超过 {max_bytes / (1024 * 1024):g} MiB。")
    return data, sha256(data).hexdigest()


def _srgb(image: Image.Image) -> Image.Image:
    has_alpha = any(band.upper() == "A" for band in image.getbands()) or "transparency" in image.info
    mode = "RGBA" if has_alpha else "RGB"
    profile = image.info.get("icc_profile")
    if profile and image.mode in {"RGB", "RGBA", "CMYK", "L", "LAB"}:
        try:
            image = ImageCms.profileToProfile(image, BytesIO(profile), ImageCms.createProfile("sRGB"), outputMode=mode)
        except (OSError, ValueError, ImageCms.PyCMSError):
            pass
    result = image.convert(mode)
    result.info.clear()
    return result


def normalize_photo(data: bytes, max_pixels: int) -> Image.Image:
    try:
        with Image.open(BytesIO(data), formats=sorted(FORMATS)) as candidate:
            if candidate.width * candidate.height > max_pixels:
                raise AppError("IMAGE_PIXELS", f"图片超过 {max_pixels:,} 像素。")
            if getattr(candidate, "n_frames", 1) != 1:
                raise AppError("IMAGE_FRAMES", "请选择单帧图片。")
            candidate.verify()
        with Image.open(BytesIO(data), formats=sorted(FORMATS)) as source:
            return _srgb(ImageOps.exif_transpose(source))
    except Image.DecompressionBombError as exc:
        raise AppError("IMAGE_PIXELS", "图片尺寸过大。") from exc
    except (OSError, SyntaxError, ValueError, OverflowError) as exc:
        raise AppError("IMAGE_DECODE", "图片已损坏或格式不受支持。") from exc


def png_bytes(image: Image.Image) -> bytes:
    with BytesIO() as stream:
        image.save(stream, format="PNG", compress_level=3)
        return stream.getvalue()


def verify_png(path: Path) -> None:
    with Image.open(path, formats=["PNG"]) as image:
        image.verify()


def on_white(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")
