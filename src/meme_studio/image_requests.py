"""Build and persist a private snapshot of one static generation request."""

import json
import os
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from .constraints import ImageRequirements, constraint_document, image_requirements
from .errors import AppError
from .provider_image import ImageGeometry, PREP_VERSION, prepare_image
from .schemas import Version
from .versions import ProjectStore, document


def compile_image_prompt(snapshot: Version, instruction: str, local: bool) -> tuple[str, ImageRequirements]:
    instructions = instruction.strip()
    if not instructions:
        raise AppError("PROMPT_EMPTY", "请填写希望如何修改图片。")
    requirements = image_requirements(snapshot, instructions)
    prompt = instructions
    if requirements.applied:
        prompt += "\n保留要求：" + "；".join(entry.text for entry in requirements.applied)
    prompt += "\n范围：仅修改白色蒙版所指区域。" if local else "\n范围：整张图片。"
    prompt += "\n输出无字底图。不要添加字幕、文字或水印。"
    if len(prompt) > 800:
        raise AppError("PROMPT_LENGTH", "修改说明与保留要求合计超过模型的 800 字符限制。")
    return prompt, requirements


def freeze_request(
    projects: ProjectStore, snapshot: Version, model: str, instruction: str, local: bool, protect: bool,
) -> tuple[str, bytes, bytes | None]:
    prompt, requirements = compile_image_prompt(snapshot, instruction, local)
    asset = projects.asset(snapshot.base_image_asset_id)
    image = projects.load_image(asset)
    region = projects.region(snapshot) if local else None
    if local and region is None:
        raise AppError("REGION_REQUIRED", "请先保存一个局部选区，再提交蒙版编辑。")
    if protect and not local:
        raise AppError("PROTECT_MODE", "保留选区外像素仅用于局部蒙版编辑。")
    mask = projects.load_image(projects.asset(region.mask_asset_id)) if region is not None else None
    image_bytes, mask_bytes, geometry = prepare_image(image, mask)
    request = document(
        preprocess=PREP_VERSION, model=model, prompt=prompt, instruction=instruction.strip(),
        function="description_edit_with_mask" if local else "description_edit",
        parameters={"n": 1, "watermark": False},
        source_version_id=snapshot.id, base_asset_id=asset.id, base_sha256=asset.sha256,
        region_id=region.id if region is not None else None,
        mask_asset_id=region.mask_asset_id if region is not None else None,
        constraints=json.loads(snapshot.constraints_json),
        applied_constraints=[asdict(entry) for entry in requirements.applied],
        replaced_constraints=[asdict(entry) for entry in requirements.replaced],
        result_constraints_json=constraint_document(requirements.remaining),
        protect=protect, geometry=asdict(geometry),
        base_sha=sha256(image_bytes).hexdigest(),
        mask_sha=sha256(mask_bytes).hexdigest() if mask_bytes is not None else None,
    )
    return request, image_bytes, mask_bytes


def persist_request(runtime: Path, job_id: str, request: str, base: bytes, mask: bytes | None) -> Path:
    parent = runtime / "jobs"
    if parent.resolve() != parent:
        raise AppError("JOB_PATH", "任务目录不能重定向到其他位置。")
    directory = parent / job_id
    created = False
    try:
        directory.mkdir(parents=True, exist_ok=False)
        created = True
        for name, data in (("base.png", base), ("mask.png", mask), ("request.json", request.encode("utf-8"))):
            if data is None:
                continue
            with (directory / name).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
    except OSError as exc:
        if created:
            discard_request(directory)
        raise AppError("JOB_WRITE", "无法保存本机任务副本，请检查磁盘空间和权限。") from exc
    return directory / "request.json"


def discard_request(directory: Path) -> None:
    if directory.is_dir() and directory.resolve() == directory:
        for name in ("base.png", "mask.png", "request.json"):
            (directory / name).unlink(missing_ok=True)
        directory.rmdir()


def read_request(runtime: Path, job) -> tuple[dict, Path]:
    path = (runtime / job["request_path"]).resolve()
    if path != runtime / "jobs" / job["id"] / "request.json":
        raise AppError("JOB_PATH", "保存的请求路径无效。")
    try:
        data = path.read_bytes()
        if sha256(data).hexdigest() != job["request_hash"]:
            raise ValueError("request hash")
        frozen = json.loads(data)
        if frozen["source_version_id"] != job["source_version_id"]:
            raise ValueError("version mismatch")
        return frozen, path.parent
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AppError("JOB_INPUT", "冻结的请求副本缺失或损坏。") from exc


def frozen_bytes(directory: Path, name: str, digest: str) -> bytes:
    path = directory / name
    try:
        if path.resolve() != path:
            raise ValueError("unsafe path")
        data = path.read_bytes()
        if sha256(data).hexdigest() != digest:
            raise ValueError("image hash")
        return data
    except (OSError, ValueError) as exc:
        raise AppError("JOB_INPUT", "冻结的图像副本缺失或损坏。") from exc
