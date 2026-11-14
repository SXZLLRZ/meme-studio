"""Local interface for the gallery and editor, with controlled media paths."""

import os

from ..config import Config
from ..errors import AppError
from ..dashscope import DashScope
from ..gallery import GalleryStore
from ..intent import IntentParser
from ..model_manager import ModelManager, ModelSlot
from ..segmentation import SamSegmenter
from ..static_jobs import StaticJobs
from ..versions import ProjectStore
from ..video_jobs import VideoJobs
from ..video_media import Ffmpeg
from ..startup import StartupStatus


def build_page(config: Config, status: StartupStatus, jobs: StaticJobs, video_jobs: VideoJobs | None = None):
    import gradio as gr

    from .editor_panel import editor_panel
    from .gallery_panel import gallery_panel
    from .video_panel import video_panel

    if video_jobs is None:
        video_jobs = VideoJobs(jobs.projects, jobs.provider, config.api.video_model, Ffmpeg(config.root), config.video)

    slot = ModelSlot()
    models = config.root / "models"
    parser = IntentParser(config.api.text_base_url, config.api.text_model, config.secrets.dashscope_api_key)

    with gr.Blocks(title="个人图库表情包工作台", analytics_enabled=False) as page:
        gr.Markdown(
            "# 个人图库表情包工作台\n"
            "**个人图库、静态编辑与动态表情包**\n\n"
            "从图库选图，编辑画面与动作，并在本机导出表情包。"
        )
        with gr.Tabs() as tabs:
            with gr.Tab("图库与检索", id="gallery"):
                gallery, selected, create, pending = gallery_panel(config, ModelManager(models, slot), parser)
            with gr.Tab("编辑与历史", id="edit"):
                editor = editor_panel(config, page, tabs, gallery, selected, create, pending, jobs,
                                      SamSegmenter(models, slot), parser)
            with gr.Tab("动态与导出", id="video") as dynamic:
                video_panel(page, dynamic, video_jobs, editor)
            with gr.Tab("本地状态", id="status"):
                key_status = "已配置" if status.api_key_present else "未配置，可使用本地功能"
                gr.Markdown(
                    f"- Python {status.python_version}\n"
                    f"- SQLite {status.sqlite_version}，数据库版本 {status.schema_version}\n"
                    f"- API Key {key_status}\n"
                    f"- 本机视频工具 {'可用' if video_jobs.media.ready else '待配置'}\n"
                    "- 静态和动态生成由用户明确提交。本机配字、选区和导出独立运行。"
                )
                gr.Textbox(label="项目数据目录", value=str(config.runtime_dir), interactive=False)
    return page


def launch(config: Config, status: StartupStatus, port: int | None, open_browser: bool) -> None:
    projects = ProjectStore(GalleryStore(config.runtime_dir))
    endpoint, key = config.api.media_base_url, config.secrets.dashscope_api_key
    jobs = StaticJobs(projects, DashScope(endpoint, key), config.api.image_model)
    video_jobs = VideoJobs(projects, DashScope(endpoint, key), config.api.video_model, Ffmpeg(config.root), config.video)
    try:
        _launch(config, status, port, open_browser, jobs, video_jobs)
    finally:
        jobs.close()
        video_jobs.close()


def _launch(config: Config, status: StartupStatus, port: int | None, open_browser: bool,
            jobs: StaticJobs, video_jobs: VideoJobs) -> None:
    page = build_page(config, status, jobs, video_jobs)
    os.environ.pop("GRADIO_ALLOWED_PATHS", None)
    published = [config.runtime_dir / name for name in ("thumbnails", "exports")]
    if any(path.resolve() != path for path in published):
        raise AppError("MEDIA_DIRECTORY", "缩略图或导出目录不能重定向到其他位置。")
    blocked = [
        config.root / ".env",
        config.runtime_dir / "app.sqlite3",
        config.runtime_dir / "logs",
        config.runtime_dir / "api_smoke",
        config.runtime_dir / "assets",
        config.runtime_dir / "jobs",
        config.root / "models",
        config.runtime_dir / "p5-sam-archive",
        config.root / "runtime" / "ffmpeg",
        config.runtime_dir / "video-work",
        config.runtime_dir / "p6-p0-video.mp4",
    ]
    jobs.start()
    video_jobs.start()
    page.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=open_browser,
        strict_cors=True,
        allowed_paths=[str(path) for path in published],
        blocked_paths=[str(path) for path in blocked],
        max_file_size=config.max_image_bytes,
        mcp_server=False,
        enable_monitoring=False,
    )
