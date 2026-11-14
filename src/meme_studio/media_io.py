"""Validate and atomically publish files in the managed asset directory."""

import hashlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from .errors import AppError


@dataclass(frozen=True)
class StoredFile:
    relative_path: str
    sha256: str
    byte_size: int


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def store_asset(
    runtime_dir: Path,
    data: bytes,
    extension: str,
    validate: Callable[[Path], None],
    *,
    folder: str = "assets",
) -> StoredFile:
    if folder not in {"assets", "thumbnails", "exports"}:
        raise AppError("ASSET_DIRECTORY", "资产目录不受支持。")
    if not data or not re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
        raise AppError("ASSET_INPUT", "资产内容为空或扩展名无效。")

    root = runtime_dir.resolve()
    assets = root / folder
    if assets.resolve() != assets:
        raise AppError("ASSET_DIRECTORY", "资产目录不能重定向到其他位置。")
    temporary: Path | None = None
    try:
        assets.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=assets, prefix=".pending-", suffix=extension, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

        validate(temporary)
        digest = file_sha256(temporary)
        destination = assets / f"{digest}{extension}"
        if destination.is_symlink():
            raise AppError("ASSET_PATH", "资产目标路径不能是符号链接。")
        if destination.is_file() and file_sha256(destination) == digest:
            return StoredFile(destination.relative_to(root).as_posix(), digest, temporary.stat().st_size)
        size = temporary.stat().st_size
        temporary.replace(destination)
        return StoredFile(destination.relative_to(root).as_posix(), digest, size)
    except OSError as exc:
        raise AppError("ASSET_WRITE", "无法校验或保存资产，请检查文件和项目目录。") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
