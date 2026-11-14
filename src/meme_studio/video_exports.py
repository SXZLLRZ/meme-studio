
"""Local video captions, MP4/GIF encoding, and immutable export cache."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from .dashscope import MAX_VIDEO_BYTES
from .database import connect
from .errors import AppError
from .schemas import Asset, Version, new_id, utc_now
from .text_layers import RENDERER_VERSION, TextLayer, text_document
from .video_media import Ffmpeg, VideoInfo, save_video, video_path
from .versions import document


EXPORT_VERSION = "p6-overlay-x264-palette-v1"


@dataclass(frozen=True)
class GifOptions:
    edge: int = 512
    fps: int = 12
    target_mib: float = 5
    seconds: float = 0

    def __post_init__(self):
        if (type(self.edge) is not int or type(self.fps) is not int
                or not 128 <= self.edge <= 720 or not 6 <= self.fps <= 20
                or not 1 <= self.target_mib <= 20 or not 0 <= self.seconds <= 15):
            raise AppError("GIF_OPTIONS", "GIF 长边须为 128–720，帧率 6–20，目标 1–20 MiB，截取时长 0–15 秒。")

    def sizes(self):
        return tuple(dict.fromkeys(
            (max(128, round(self.edge * edge)), max(6, round(self.fps * fps)))
            for edge, fps in ((1, 1), (.82, 1), (.70, .82), (.55, .66))
        ))


@dataclass(frozen=True)
class VideoExport:
    path: Path
    info: VideoInfo
    byte_size: int
    cached: bool
    target_mib: float | None = None

    def label(self) -> str:
        summary = f"{self.info.label()} · {self.byte_size / 1024**2:.2f} MiB"
        if self.target_mib is not None and self.byte_size > self.target_mib * 1024**2:
            summary += f" · 超过 {self.target_mib:g} MiB 目标，可下载当前文件或缩短片段后再导出。"
        return ("已复用缓存 · " if self.cached else "已在本机导出 · ") + summary


class VideoExports:
    def __init__(self, projects, renderer, media: Ffmpeg):
        self.projects = projects
        self.renderer = renderer
        self.media = media
        self._lock = Lock()

    def source(self, snapshot: Version) -> tuple[Asset, Path, VideoInfo]:
        if not snapshot.base_video_asset_id:
            raise AppError("VIDEO_REQUIRED", "请打开一个已保存的视频版本。")
        asset = self.projects.asset(snapshot.base_video_asset_id)
        path = video_path(self.projects, asset)
        return asset, path, self.media.probe(path)

    def export(self, snapshot: Version, gif: GifOptions | None = None) -> VideoExport:
        with self._lock:
            self.projects.require_current(snapshot.project_id, snapshot.id)
            return self._export(snapshot, gif)

    def _export(self, snapshot: Version, gif: GifOptions | None) -> VideoExport:
        asset, source, info = self.source(snapshot)
        layers = tuple(TextLayer(**layer) for layer in json.loads(snapshot.text_layers_json)["layers"])
        fonts = self.renderer.resolve_fonts(layers)
        key = document(
            exporter=EXPORT_VERSION, text_renderer=RENDERER_VERSION, source_sha=asset.sha256,
            text=text_document(layers), fonts=[font.digest for font in fonts], gif=asdict(gif) if gif else None,
        )
        options = document(cache_key=sha256(key.encode("utf-8")).hexdigest())
        format_name = "gif" if gif else "mp4"
        with connect(self.projects.database) as db:
            row = db.execute(
                "SELECT asset_id FROM exports WHERE version_id = ? AND format = ? AND options_json = ?",
                (snapshot.id, format_name, options),
            ).fetchone()
        target = gif.target_mib if gif else None
        if row is not None:
            try:
                saved = self.projects.asset(row["asset_id"])
                path = video_path(self.projects, saved)
                return VideoExport(path, self.media.probe(path, gif=bool(gif)), saved.byte_size, True, target)
            except AppError:
                pass

        scratch = self.projects.runtime / "video-work"
        if scratch.resolve() != scratch:
            raise AppError("VIDEO_PATH", "视频工作目录不能重定向到其他位置。")
        try:
            scratch.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="export-", dir=scratch) as name:
                work = Path(name)
                overlay = None
                if layers:
                    overlay = work / "caption.png"
                    self.renderer.render_overlay((info.width, info.height), layers, fonts).save(overlay)
                result = work / f"output.{format_name}"
                if gif:
                    self._gif(source, overlay, result, info, gif)
                else:
                    self._mp4(source, overlay, result)
                if result.stat().st_size > MAX_VIDEO_BYTES:
                    raise AppError("VIDEO_EXPORT_SIZE", "导出文件超过 96 MiB，请缩短片段或降低尺寸。")
                saved, encoded = save_video(self.projects, self.media, result.read_bytes(), export=True, gif=bool(gif))
        except OSError as exc:
            raise AppError("VIDEO_EXPORT", "无法保存导出文件，请检查项目目录和磁盘空间。") from exc

        with connect(self.projects.database) as db:
            db.execute(
                """INSERT INTO exports (id, version_id, format, options_json, asset_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(version_id, format, options_json)
                   DO UPDATE SET asset_id = excluded.asset_id""",
                (new_id(), snapshot.id, format_name, options, saved.id, utc_now()),
            )
        return VideoExport(video_path(self.projects, saved), encoded, saved.byte_size, False, target)

    @staticmethod
    def _inputs(source: Path, overlay: Path | None, seconds: float = 0) -> tuple[list[str], str]:
        trim = ["-t", str(seconds)] if seconds else []
        inputs = ["-protocol_whitelist", "file", "-threads", "2", "-f", "mov", "-enable_drefs", "0", *trim,
                  "-i", str(source)]
        if overlay is None:
            return inputs, "[0:V:0]"
        inputs.extend(["-protocol_whitelist", "file", "-i", str(overlay)])
        return inputs, "[0:V:0][1:v]overlay=eof_action=repeat:format=auto,"

    def _mp4(self, source: Path, overlay: Path | None, result: Path) -> None:
        inputs, graph = self._inputs(source, overlay)
        graph += "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p[v]"
        self.media.encode([
            *inputs, "-filter_complex", graph, "-map", "[v]", "-map", "0:a:0?",
            "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-movflags", "+faststart", str(result),
        ])

    def _gif(self, source: Path, overlay: Path | None, result: Path, info: VideoInfo, options: GifOptions) -> None:
        inputs, graph = self._inputs(source, overlay, options.seconds)
        palette = result.with_name("palette.png")
        for edge, fps in options.sizes():
            ratio = min(1.0, edge / max(info.width, info.height))
            width, height = (max(1, round(dimension * ratio)) for dimension in (info.width, info.height))
            reduced = graph + f"scale={width}:{height}:flags=lanczos,setsar=1,fps={fps}"
            self.media.encode([
                *inputs, "-filter_complex", reduced + ",palettegen=stats_mode=diff[p]",
                "-map", "[p]", "-frames:v", "1", "-threads", "1", "-update", "1", str(palette),
            ])
            palette_index = 2 if overlay else 1
            self.media.encode([
                *inputs, "-protocol_whitelist", "file", "-i", str(palette), "-filter_complex",
                reduced + f"[scaled];[scaled][{palette_index}:v]paletteuse=dither=bayer:bayer_scale=3[g]",
                "-map", "[g]", "-an", "-loop", "0", str(result),
            ])
            if result.stat().st_size <= options.target_mib * 1024**2:
                break
