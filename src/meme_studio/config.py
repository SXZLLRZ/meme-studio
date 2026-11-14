"""Configuration shared by startup and future service modules."""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from .errors import AppError


@dataclass(frozen=True)
class ApiSettings:
    text_base_url: str
    media_base_url: str
    text_model: str
    image_model: str
    video_model: str


@dataclass(frozen=True)
class VideoDefaults:
    resolution: str
    duration: int
    audio: bool


@dataclass(frozen=True)
class Secrets:
    dashscope_api_key: str = field(repr=False)

    @property
    def key_present(self) -> bool:
        return bool(self.dashscope_api_key)


@dataclass(frozen=True)
class Config:
    root: Path
    runtime_dir: Path
    gallery_dir: Path
    api: ApiSettings
    video: VideoDefaults
    secrets: Secrets = field(repr=False)
    max_image_bytes: int = 24 * 1024**2
    max_image_pixels: int = 24_000_000
    font_path: Path | None = None


def _path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (root / path).resolve()


def load_config(root: Path, environ: Mapping[str, str] | None = None) -> Config:
    root = root.resolve()
    try:
        settings = json.loads((root / "settings.json").read_text(encoding="utf-8-sig"))
        file_values = dotenv_values(root / ".env", encoding="utf-8-sig", interpolate=False)
        values = {**file_values, **(os.environ if environ is None else environ)}

        def setting(env_name: str, default: str) -> str:
            value = values.get(env_name, default)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(env_name)
            return value.strip()

        runtime_dir = _path(root, setting("MEME_STUDIO_RUNTIME_DIR", settings["runtime_dir"]))
        if runtime_dir == root or not runtime_dir.is_relative_to(root):
            raise AppError("RUNTIME_PATH", "运行数据目录必须是项目目录内的子目录。")
        gallery_dir = _path(root, setting("MEME_STUDIO_GALLERY_DIR", settings["gallery_dir"]))
        limits = settings.get("gallery_limits", {})
        if not isinstance(limits, dict):
            raise ValueError("gallery_limits")
        max_mib = int(setting("MEME_STUDIO_MAX_IMAGE_MIB", str(limits.get("max_image_mib", 24))))
        max_pixels = int(setting("MEME_STUDIO_MAX_IMAGE_PIXELS", str(limits.get("max_image_pixels", 24_000_000))))
        if not 1 <= max_mib <= 128 or not 1 <= max_pixels <= 50_000_000:
            raise ValueError("gallery_limits")
        font = values.get("MEME_STUDIO_FONT_PATH", settings.get("editor_font", ""))
        if font is not None and not isinstance(font, str):
            raise ValueError("editor_font")
        api = settings["api"]
        video = settings["video_defaults"]
        if type(video["duration"]) is not int or video["duration"] <= 0 or type(video["audio"]) is not bool:
            raise ValueError("video_defaults")
        return Config(
            root=root,
            runtime_dir=runtime_dir,
            gallery_dir=gallery_dir,
            api=ApiSettings(
                text_base_url=setting("MEME_STUDIO_TEXT_BASE_URL", api["text_base_url"]).rstrip("/"),
                media_base_url=setting("MEME_STUDIO_MEDIA_BASE_URL", api["media_base_url"]).rstrip("/"),
                text_model=setting("MEME_STUDIO_TEXT_MODEL", api["text_model"]),
                image_model=setting("MEME_STUDIO_IMAGE_MODEL", api["image_model"]),
                video_model=setting("MEME_STUDIO_VIDEO_MODEL", api["video_model"]),
            ),
            video=VideoDefaults(
                resolution=setting("MEME_STUDIO_VIDEO_RESOLUTION", video["resolution"]),
                duration=video["duration"],
                audio=video["audio"],
            ),
            secrets=Secrets((values.get("DASHSCOPE_API_KEY") or "").strip()),
            max_image_bytes=max_mib * 1024**2,
            max_image_pixels=max_pixels,
            font_path=_path(root, font.strip()) if font and font.strip() else None,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise AppError("CONFIG", "无法读取项目配置。请检查 settings.json 和本地环境配置。") from exc
