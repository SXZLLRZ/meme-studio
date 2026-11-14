"""Local startup and opt-in diagnostics. No provider requests are made here."""

import importlib
import importlib.metadata
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .database import initialize
from .errors import AppError
from .logging_setup import configure_logging
from .video_media import Ffmpeg


REQUIRED = (
    ("gradio", "gradio"),
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
    ("fonttools", "fontTools"),
    ("httpx", "httpx"),
    ("python-dotenv", "dotenv"),
)
OPTIONAL = ("torch", "torchvision", "cn-clip", "SAM-2")


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class StartupStatus:
    python: str
    python_version: str
    sqlite_version: str
    schema_version: int
    required: dict[str, str]
    optional: dict[str, str | None]
    gallery_directory_exists: bool
    api_key_present: bool
    video_tools: dict[str, str | bool | None]


def prepare(config: Config) -> StartupStatus:
    # A separate cache keeps any framework-generated files out of the system temp tree.
    cache = config.runtime_dir / "gradio-cache"
    if cache.resolve() != cache:
        raise AppError("GRADIO_CACHE", "本地界面缓存不能重定向到其他位置。")
    os.environ["GRADIO_TEMP_DIR"] = str(cache)
    os.environ["GRADIO_EXAMPLES_CACHE"] = str(cache)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    required = {}
    for distribution, module in REQUIRED:
        try:
            importlib.import_module(module)
            required[distribution] = importlib.metadata.version(distribution)
        except (ImportError, OSError) as exc:
            raise AppError("DEPENDENCY", f"无法加载必需依赖 {distribution}。请检查项目独立环境。") from exc

    database_path = config.runtime_dir / "app.sqlite3"
    if database_path.is_symlink():
        raise AppError("DATABASE_PATH", "项目数据库不能是符号链接。")
    schema_version = initialize(database_path)
    logger = configure_logging(config.runtime_dir / "logs", config.secrets.dashscope_api_key)
    logger.info("本地环境检查通过，数据库版本 %s", schema_version)
    video_tools = Ffmpeg(config.root).versions()
    return StartupStatus(
        python=sys.executable,
        python_version=sys.version.split()[0],
        sqlite_version=sqlite3.sqlite_version,
        schema_version=schema_version,
        required=required,
        optional={**{package: _version(package) for package in OPTIONAL},
                  **{tool: video_tools[tool] for tool in ("ffmpeg", "ffprobe")}},
        gallery_directory_exists=config.gallery_dir.is_dir(),
        api_key_present=config.secrets.key_present,
        video_tools=video_tools,
    )


def check_gpu() -> dict:
    try:
        import torch
        import torchvision

        if not torch.cuda.is_available():
            raise AppError("CUDA_UNAVAILABLE", "项目环境未检测到可用 CUDA 显卡，本地页面仍可启动。")
        values = torch.arange(1, 5, device="cuda", dtype=torch.float32)
        result = values.square().sum().item()
        boxes = torch.tensor([[0, 0, 2, 2], [3, 3, 5, 5]], device="cuda", dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], device="cuda", dtype=torch.float32)
        kept = torchvision.ops.nms(boxes, scores, 0.5).tolist()
        if result != 30 or kept != [0, 1]:
            raise AppError("CUDA_RESULT", "显卡验证结果异常。")
        return {
            "device": torch.cuda.get_device_name(0),
            "memory_mib": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "sum_of_squares": result,
            "torchvision_nms": kept,
        }
    except (ImportError, OSError, RuntimeError) as exc:
        raise AppError("CUDA_CHECK", "CUDA 检查失败，请检查项目环境的 PyTorch、torchvision 和驱动。") from exc


def render_diagnostics(status: StartupStatus, gpu: dict | None = None) -> str:
    report = asdict(status)
    if gpu is not None:
        report["gpu"] = gpu
    return json.dumps(report, ensure_ascii=False, indent=2)
