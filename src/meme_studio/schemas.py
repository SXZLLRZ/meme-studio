"""Identifiers and data contracts for the gallery, editing and export stages."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4


def new_id() -> str:
    return str(uuid4())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AssetKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    MASK = "mask"
    TEXT_OVERLAY = "text_overlay"
    EXPORT = "export"


class VersionKind(StrEnum):
    ROOT = "root"
    IMAGE_EDIT = "image_edit"
    TEXT_EDIT = "text_edit"
    REGION_EDIT = "region_edit"
    CONSTRAINT_EDIT = "constraint_edit"
    VIDEO = "video"


class JobKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class JobStatus(StrEnum):
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    SUBMISSION_UNKNOWN = "SUBMISSION_UNKNOWN"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    REMOTE_SUCCEEDED = "REMOTE_SUCCEEDED"
    DOWNLOADING = "DOWNLOADING"
    PAUSED = "PAUSED"
    RESULT_EXPIRED = "RESULT_EXPIRED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class IntentAction(StrEnum):
    SEARCH = "search"
    SEARCH_THEN_EDIT = "search_then_edit"
    EDIT_IMAGE = "edit_image"
    EDIT_TEXT = "edit_text"
    GENERATE_VIDEO = "generate_video"
    EDIT_VIDEO_INSTRUCTION = "edit_video_instruction"
    UPDATE_CONSTRAINTS = "update_constraints"
    UNDO = "undo"
    CHECKOUT = "checkout"
    EXPORT = "export"
    CLARIFY = "clarify"


class ExportFormat(StrEnum):
    PNG = "png"
    MP4 = "mp4"
    GIF = "gif"


@dataclass(frozen=True)
class Asset:
    id: str
    sha256: str
    relative_path: str
    mime: str
    byte_size: int
    kind: AssetKind
    created_at: str
    width: int | None = None
    height: int | None = None
    duration: float | None = None


@dataclass(frozen=True)
class Project:
    id: str
    title: str
    source_image_id: str
    root_version_id: str | None
    current_version_id: str | None
    created_at: str
    updated_at: str
    revision: int = 0


@dataclass(frozen=True)
class Version:
    id: str
    project_id: str
    kind: VersionKind
    base_image_asset_id: str
    created_at: str
    parent_id: str | None = None
    base_video_asset_id: str | None = None
    anchor_image_version_id: str | None = None
    text_layers_json: str = '{"schema_version":1,"layers":[]}'
    constraints_json: str = '{"schema_version":1,"constraints":[]}'
    operation_json: str = '{"schema_version":1}'
    job_id: str | None = None
    region_id: str | None = None


@dataclass(frozen=True)
class Region:
    id: str
    project_id: str
    source_version_id: str
    base_asset_id: str
    base_asset_hash: str
    width: int
    height: int
    mask_asset_id: str
    prompts_json: str
    transform_json: str
    created_at: str


@dataclass(frozen=True)
class Job:
    id: str
    project_id: str
    source_version_id: str
    kind: JobKind
    local_status: JobStatus
    provider: str
    model: str
    request_hash: str
    request_path: str
    created_at: str
    updated_at: str
    remote_status: str | None = None
    remote_task_id: str | None = None
    result_path: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class IntentPlan:
    action: IntentAction
    search_query: str = ""
    image_instruction: str = ""
    video_instruction: str = ""
    caption: str = ""
    region_required: bool = False
    constraint_add: tuple[dict[str, str], ...] = ()
    constraint_remove: tuple[str, ...] = ()
    needs_clarification: bool = False
    clarification_text: str = ""
    schema_version: int = 1


@dataclass(frozen=True)
class Export:
    id: str
    version_id: str
    format: ExportFormat
    options_json: str
    asset_id: str
    created_at: str
