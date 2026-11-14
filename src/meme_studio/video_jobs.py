
"""Persistent image-to-video jobs using a frozen static anchor."""

from hashlib import sha256

from PIL import Image

from .config import VideoDefaults
from .constraints import constraint_document, read_constraints, remaining_after
from .errors import AppError
from .image_files import on_white, png_bytes
from .image_requests import read_request
from .provider_image import MAX_INPUT_BYTES
from .schemas import JobKind
from .static_jobs import MediaJobs
from .video_media import Ffmpeg, save_video
from .versions import document


VIDEO_MODEL = "wan2.6-i2v-flash"


def prepare_frame(image):
    width, height = image.size
    scale = min(1440 / max(image.size), max(1, 240 / min(image.size)))
    size = tuple(max(1, round(edge * scale)) for edge in (width, height))
    clean = on_white(image)
    if size != clean.size:
        clean = clean.resize(size, Image.Resampling.LANCZOS)
    minimum = max(240, (max(size) + 3) // 4)
    canvas_size = tuple(max(edge, minimum) for edge in size)
    offset = tuple((out - inner) // 2 for out, inner in zip(canvas_size, size))
    canvas = Image.new("RGB", canvas_size, "white")
    canvas.paste(clean, offset)
    data = png_bytes(canvas)
    if len(data) > MAX_INPUT_BYTES:
        raise AppError("VIDEO_FRAME_SIZE", "准备后的首帧超过项目请求上限。")
    return data, dict(source_size=image.size, resized=size, canvas=canvas_size, offset=offset, background="#FFFFFF")


def anchor_for(projects, source):
    if source.base_video_asset_id is None:
        return source
    if source.anchor_image_version_id is None:
        raise AppError("VIDEO_ANCHOR", "这个视频版本缺少静态锚点。")
    anchor = projects.version(source.project_id, source.anchor_image_version_id)
    if anchor.base_video_asset_id is not None or anchor.base_image_asset_id != source.base_image_asset_id:
        raise AppError("VIDEO_ANCHOR", "这个视频版本的静态锚点无效。")
    return anchor


class VideoJobs(MediaJobs):
    kind = JobKind.VIDEO
    supported_model = VIDEO_MODEL

    def __init__(self, projects, provider, model: str, media: Ffmpeg, defaults: VideoDefaults, **kwargs):
        super().__init__(projects, provider, model, **kwargs)
        self.media = media
        self.defaults = defaults

    @property
    def ready(self) -> bool:
        return super().ready and self.media.ready

    def submit(self, project_id: str, expected_id: str, instruction: str) -> str:
        if not self.ready:
            raise AppError("VIDEO_CONFIG", "生成视频需要本机视频工具及已验证的媒体密钥和模型。")
        source = self.projects.require_current(project_id, expected_id)
        anchor = anchor_for(self.projects, source)
        instruction = instruction.strip()
        if not instruction:
            raise AppError("VIDEO_PROMPT", "请填写希望主体如何运动。")
        settings = self.defaults
        if settings.resolution not in {"720P", "1080P"} or not 2 <= settings.duration <= 15:
            raise AppError("VIDEO_OPTIONS", "视频需使用 720P 或 1080P，时长为 2 至 15 秒。")
        entries = read_constraints(source)
        applied = tuple(entry for entry in entries if entry.applies("video"))
        prompt = instruction
        if applied:
            prompt += "\n动态保留要求：" + "；".join(entry.text for entry in applied)
        prompt += "\n保持单个连续镜头，输出无字视频，不加字幕或水印。"
        if len(prompt) > 1500:
            raise AppError("VIDEO_PROMPT", "动作描述和动态保留要求合计最多 1500 字。")
        asset = self.projects.asset(anchor.base_image_asset_id)
        base, geometry = prepare_frame(self.projects.load_image(asset))
        request = document(
            model=self.model, prompt=prompt, instruction=instruction, function="image_to_video",
            parameters=dict(resolution=settings.resolution, duration=settings.duration, audio=settings.audio,
                            prompt_extend=False, watermark=False),
            anchor_image_version_id=anchor.id, source_version_id=source.id, base_asset_id=asset.id,
            base_sha256=asset.sha256, base_sha=sha256(base).hexdigest(), geometry=geometry,
            result_constraints_json=constraint_document(remaining_after(entries, "video")),
        )
        return self._queue(source, request, base)

    def _submit_frozen(self, frozen, base, mask):
        return self.provider.submit_video(frozen, base)

    def _download(self, job) -> None:
        frozen, _ = read_request(self.projects.runtime, job)
        source = self.projects.version(job["project_id"], job["source_version_id"])
        anchor = anchor_for(self.projects, source)
        asset = self.projects.asset(anchor.base_image_asset_id)
        if (anchor.id, asset.id, asset.sha256) != (
            frozen["anchor_image_version_id"], frozen["base_asset_id"], frozen["base_sha256"],
        ):
            raise AppError("JOB_INPUT", "保存的首帧与静态锚点不一致。")
        data = self.provider.download_video(job["result_url"])
        try:
            output, _ = save_video(self.projects, self.media, data)
        except AppError as exc:
            if exc.code in {"ASSET_WRITE", "VIDEO_TOOLS", "VIDEO_READ"}:
                raise AppError("RESULT_SAVE", "视频已生成，本机暂时无法保存，可继续下载。", retryable=True) from exc
            raise
        self._record_result(job, source, output, frozen)
