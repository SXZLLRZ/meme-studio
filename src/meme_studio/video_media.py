
"""Probe bounded local media and invoke the project FFmpeg without a shell."""

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path

from .dashscope import MAX_VIDEO_BYTES
from .errors import AppError
from .media_io import file_sha256, store_asset
from .schemas import Asset, AssetKind


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float
    frames: int | None
    audio: bool
    codec: str

    def label(self) -> str:
        frames = f" · {self.frames} 帧" if self.frames is not None else ""
        return f"{self.width}×{self.height} · {self.duration:.2f} 秒 · {self.fps:.2f} fps{frames} · {'有音轨' if self.audio else '无音轨'}"


def _locate(root: Path, name: str) -> Path | None:
    filename = name + (".exe" if os.name == "nt" else "")
    bundled = root / "runtime" / "ffmpeg" / "bin" / filename
    if bundled.is_file():
        return bundled.resolve()
    found = shutil.which(name)
    return Path(found).resolve() if found else None


class Ffmpeg:
    def __init__(self, root: Path):
        self.ffmpeg = _locate(root, "ffmpeg")
        self.ffprobe = _locate(root, "ffprobe")

    def _call(self, program: Path | None, arguments: list[str], *, timeout: int, code: str, message: str) -> str:
        if program is None:
            raise AppError("VIDEO_TOOLS", "请在项目目录安装 FFmpeg 与 ffprobe 后重试。")
        try:
            result = subprocess.run(
                [str(program), *arguments], stdin=subprocess.DEVNULL, capture_output=True,
                check=True, timeout=timeout, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise AppError("VIDEO_TOOLS", "无法启动本机 FFmpeg 或 ffprobe。") from exc
        except (subprocess.SubprocessError, ValueError) as exc:
            raise AppError(code, message) from exc
        return result.stdout

    @cached_property
    def ready(self) -> bool:
        if self.ffmpeg is None or self.ffprobe is None:
            return False
        try:
            self._call(self.ffprobe, ["-version"], timeout=8, code="VIDEO_TOOLS",
                       message="无法检查 ffprobe。")
            encoders = self._call(self.ffmpeg, ["-hide_banner", "-encoders"], timeout=8,
                                  code="VIDEO_TOOLS", message="无法检查 H.264 编码器。")
        except AppError:
            return False
        return bool(re.search(r"^\s*V\S*\s+libx264\s", encoders, re.MULTILINE))

    def versions(self) -> dict[str, str | bool | None]:
        def show(executable):
            if executable is None:
                return None
            try:
                lines = self._call(executable, ["-version"], timeout=8, code="VIDEO_TOOLS",
                                   message="无法检查视频工具版本。").splitlines()
                return lines[0] if lines else None
            except AppError:
                return None
        return {
            "ffmpeg": show(self.ffmpeg), "ffprobe": show(self.ffprobe), "h264_libx264": self.ready,
            "ffmpeg_path": str(self.ffmpeg) if self.ffmpeg else None,
            "ffprobe_path": str(self.ffprobe) if self.ffprobe else None,
        }

    def probe(self, path: Path, *, gif: bool = False) -> VideoInfo:
        try:
            size = path.stat().st_size
            with path.open("rb") as stream:
                header = stream.read(12)
        except OSError as exc:
            raise AppError("VIDEO_READ", "本机视频文件无法读取。") from exc
        signature = header[:6] in {b"GIF87a", b"GIF89a"} if gif else header[4:8] == b"ftyp"
        if not 0 < size <= MAX_VIDEO_BYTES or not signature:
            raise AppError("VIDEO_FORMAT", "视频容器不符合预期，或超过本机的 96 MiB 上限。")
        fields = ("stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,duration,nb_frames,"
                  "nb_read_frames:stream_disposition=attached_pic:stream_side_data=rotation:format=duration")
        args = ["-v", "error", "-protocol_whitelist", "file", "-max_alloc", str(128 * 1024**2)]
        args += ["-f", "gif", "-count_frames"] if gif else ["-f", "mov", "-enable_drefs", "0"]
        data = self._call(self.ffprobe, [*args, "-show_entries", fields, "-of", "json", str(path)],
                          timeout=30, code="VIDEO_FORMAT", message="视频损坏或 ffprobe 无法读取。")
        try:
            probe = json.loads(data)
            streams = probe["streams"]
            video = [s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
            if len(video) != 1:
                raise ValueError
            stream = video[0]
            codec = stream["codec_name"]
            width, height = int(stream["width"]), int(stream["height"])
            if any(abs(int(side.get("rotation", 0))) % 180 == 90 for side in stream.get("side_data_list", [])):
                width, height = height, width
            duration = float(probe.get("format", {}).get("duration", stream.get("duration", 0)))
            count = stream.get("nb_read_frames", stream.get("nb_frames"))
            frames = int(count) if count not in {None, "N/A"} else None
            fps = float(Fraction(stream.get("avg_frame_rate", "0")))
            if not fps:
                fps = float(Fraction(stream.get("r_frame_rate", "0")))
            if gif and frames and duration > 0:
                fps = frames / duration
            if (
                codec != ("gif" if gif else "h264") or not 1 <= width <= 4096 or not 1 <= height <= 4096
                or width * height > 8_847_360 or not 0 < duration <= 20 or not 0 < fps <= 120
                or (frames is not None and not 1 <= frames <= 2400) or (gif and frames is None)
                or not math.isfinite(duration) or not math.isfinite(fps)
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise AppError("VIDEO_FORMAT", "视频帧率、尺寸、时长或编码超出项目支持范围。") from exc
        return VideoInfo(width, height, duration, fps, frames, any(s["codec_type"] == "audio" for s in streams), codec)

    def encode(self, arguments: list[str]) -> None:
        if not self.ready:
            raise AppError("VIDEO_TOOLS", "本机视频导出需要 FFmpeg、ffprobe 和 libx264。")
        self._call(
            self.ffmpeg, ["-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                          "-filter_threads", "2", "-filter_complex_threads", "2", *arguments],
            timeout=90, code="VIDEO_ENCODE", message="本机视频编码失败或超时，可调整参数后重试。",
        )


def video_path(projects, asset: Asset) -> Path:
    folder = "exports" if asset.kind == AssetKind.EXPORT else "assets"
    extension = {"video/mp4": ".mp4", "image/gif": ".gif"}.get(asset.mime)
    if extension is None or asset.kind not in {AssetKind.VIDEO, AssetKind.EXPORT}:
        raise AppError("VIDEO_ASSET", "该版本没有可读取的视频文件。")
    path = (projects.runtime / asset.relative_path).resolve()
    expected = projects.runtime / folder / f"{asset.sha256}{extension}"
    try:
        valid = path == expected and path.is_file() and file_sha256(path) == asset.sha256
    except OSError:
        valid = False
    if not valid:
        raise AppError("VIDEO_ASSET", "本机视频文件缺失或校验失败。")
    return path


def save_video(projects, media: Ffmpeg, data: bytes, *, export=False, gif=False):
    info = None

    def verify(path):
        nonlocal info
        info = media.probe(path, gif=gif)

    mime, extension = ("image/gif", ".gif") if gif else ("video/mp4", ".mp4")
    kind = AssetKind.EXPORT if export else AssetKind.VIDEO
    stored = store_asset(projects.runtime, data, extension, verify, folder="exports" if export else "assets")
    return projects.register_asset(stored, kind, mime, info.width, info.height, info.duration), info
