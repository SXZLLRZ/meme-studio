"""Persistent media job queue and static image job specialization."""

import logging
import threading
import time
from dataclasses import dataclass
from hashlib import sha256

from .dashscope import PROVIDER, DashScope
from .database import connect
from .errors import AppError
from .image_requests import discard_request, freeze_request, frozen_bytes, persist_request, read_request
from .provider_image import ImageGeometry, decode_result, map_result
from .schemas import AssetKind, JobKind, JobStatus as Status, new_id, utc_now
from .versions import ProjectStore


WAIT_SECONDS = 600
LEASE_SECONDS = 150
POLL_DELAYS = (3, 5, 10, 15)
ACTIVE = (Status.CREATED, Status.SUBMITTING, Status.QUEUED, Status.RUNNING, Status.REMOTE_SUCCEEDED, Status.DOWNLOADING)
READY = tuple(status for status in ACTIVE if status != Status.SUBMITTING)
LOGGER = logging.getLogger("meme_studio.static_jobs")


def _slots(values) -> str:
    return ",".join("?" for _ in values)


@dataclass(frozen=True)
class JobView:
    id: str
    source_version_id: str
    local_status: str
    remote_status: str | None
    remote_task_id: str | None
    paused: bool
    error_code: str | None
    result_version_id: str | None
    image_count: int | None
    video_seconds: float | None
    resolution: str | None
    audio: int | None


class MediaJobs:
    kind: JobKind
    supported_model: str

    def __init__(self, projects: ProjectStore, provider: DashScope, model: str, *, clock=time.time):
        self.projects = projects
        self.provider = provider
        self.model = model
        self.clock = clock
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self.provider.ready and self.model == self.supported_model

    def start(self) -> None:
        if self.ready and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name=f"{self.kind}-jobs", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._thread is None or not self._thread.is_alive():
            self.provider.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.run_once():
                    continue
            except Exception as exc:
                LOGGER.error("任务处理暂时中断：%s", type(exc).__name__)
            self._wake.wait(0.75)
            self._wake.clear()
        self.provider.close()

    def list(self, project_id: str) -> list[JobView]:
        with connect(self.projects.database) as db:
            rows = db.execute(
                """SELECT j.id, j.source_version_id, j.local_status, j.remote_status, j.remote_task_id,
                          j.paused, j.error_code, v.id AS result_version_id, u.image_count, u.video_seconds, u.resolution, u.audio
                   FROM jobs AS j LEFT JOIN versions AS v ON v.job_id = j.id
                   LEFT JOIN usage_events AS u ON u.job_id = j.id
                   WHERE j.project_id = ? AND j.provider = ? AND j.kind = ?
                   ORDER BY j.rowid DESC LIMIT 40""", (project_id, PROVIDER, self.kind),
            ).fetchall()
        return [JobView(**row) for row in rows]

    def _queue(self, source, request: str, base: bytes, mask: bytes | None = None) -> str:
        project_id, expected_id = source.project_id, source.id
        digest = sha256(request.encode("utf-8")).hexdigest()
        job_id = new_id()
        path = persist_request(self.projects.runtime, job_id, request, base, mask)
        committed = False
        try:
            with connect(self.projects.database) as db:
                db.execute("BEGIN IMMEDIATE")
                prior = db.execute(
                    """SELECT id FROM jobs WHERE project_id = ? AND source_version_id = ? AND request_hash = ?
                       AND local_status NOT IN ('FAILED', 'RESULT_EXPIRED')""", (project_id, expected_id, digest),
                ).fetchone()
                if prior is not None:
                    return prior["id"]
                project = db.execute(
                    "SELECT current_version_id, revision FROM projects WHERE id = ?", (project_id,),
                ).fetchone()
                if project is None or project["current_version_id"] != expected_id:
                    raise AppError("STALE_VERSION", "准备期间当前版本已改变，请重新打开项目。")
                active = db.execute(
                    f"SELECT 1 FROM jobs WHERE project_id = ? AND local_status IN ({_slots(ACTIVE)})",
                    (project_id, *ACTIVE),
                ).fetchone()
                if active is not None:
                    raise AppError("JOB_BUSY", "该项目已有任务，请在任务区等待或继续查询。")
                now = utc_now()
                db.execute(
                    """INSERT INTO jobs (id, project_id, source_version_id, kind, local_status, provider, model,
                       request_hash, request_path, created_at, updated_at, source_revision, wait_until)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        job_id, project_id, expected_id, self.kind, Status.CREATED, PROVIDER, self.model, digest,
                        path.relative_to(self.projects.runtime).as_posix(), now, now, project["revision"],
                        self.clock() + WAIT_SECONDS,
                    ),
                )
            committed = True
        finally:
            if not committed:
                discard_request(path.parent)
        self._wake.set()
        return job_id

    def set_paused(self, project_id: str, job_id: str, paused: bool) -> None:
        with connect(self.projects.database) as db:
            updated = db.execute(
                f"""UPDATE jobs SET paused = ?, wait_until = ?, next_action_at = 0, error_code = NULL, updated_at = ?
                    WHERE id = ? AND project_id = ? AND provider = ? AND kind = ?
                      AND local_status IN ({_slots(ACTIVE)})""",
                (int(paused), self.clock() + WAIT_SECONDS, utc_now(), job_id, project_id, PROVIDER, self.kind, *ACTIVE),
            ).rowcount
        if not updated:
            raise AppError("JOB_STATE", "请从当前项目选择一个可暂停或继续的任务。")
        self._wake.set()

    def _claim(self):
        now = self.clock()
        with connect(self.projects.database) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """UPDATE jobs SET local_status = ?, error_code = 'SUBMIT_UNKNOWN', lease_token = NULL,
                          lease_until = 0, updated_at = ?
                   WHERE provider = ? AND kind = ? AND local_status = ? AND lease_until <= ?""",
                (Status.SUBMISSION_UNKNOWN, utc_now(), PROVIDER, self.kind, Status.SUBMITTING, now),
            )
            db.execute(
                f"""UPDATE jobs SET paused = 1, error_code = 'WAIT_TIMEOUT', updated_at = ?
                    WHERE provider = ? AND kind = ? AND local_status IN ({_slots(READY)})
                      AND paused = 0 AND wait_until <= ? AND lease_until <= ?""",
                (utc_now(), PROVIDER, self.kind, *READY, now, now),
            )
            row = db.execute(
                f"""SELECT * FROM jobs WHERE provider = ? AND kind = ? AND paused = 0
                    AND local_status IN ({_slots(READY)}) AND lease_until <= ? AND next_action_at <= ?
                    ORDER BY next_action_at, rowid LIMIT 1""",
                (PROVIDER, self.kind, *READY, now, now),
            ).fetchone()
            if row is None:
                return None
            status = row["local_status"]
            if status == Status.CREATED:
                status = Status.SUBMITTING
            elif status == Status.REMOTE_SUCCEEDED:
                status = Status.DOWNLOADING
            token = new_id()
            db.execute(
                "UPDATE jobs SET local_status = ?, lease_token = ?, lease_until = ?, updated_at = ? WHERE id = ?",
                (status, token, now + LEASE_SECONDS, utc_now(), row["id"]),
            )
            return dict(row, local_status=status, lease_token=token)

    def _delay(self, job) -> float:
        return self.clock() + POLL_DELAYS[min(job["poll_count"], len(POLL_DELAYS) - 1)]

    def _change(self, db, job, **fields) -> bool:
        fields.update(lease_token=None, lease_until=0, updated_at=utc_now())
        columns = ", ".join(f"{field} = ?" for field in fields)
        return bool(db.execute(
            f"UPDATE jobs SET {columns} WHERE id = ? AND lease_token = ?",
            (*fields.values(), job["id"], job["lease_token"]),
        ).rowcount)

    def _finish(self, job, **fields) -> None:
        with connect(self.projects.database) as db:
            self._change(db, job, **fields)

    def run_once(self) -> bool:
        if not self.ready:
            return False
        job = self._claim()
        if job is None:
            return False
        try:
            if job["local_status"] == Status.SUBMITTING:
                self._send(job)
            elif job["local_status"] == Status.DOWNLOADING:
                self._download(job)
            else:
                self._poll(job)
        except AppError as exc:
            self._handle_error(job, exc)
        return True

    def _send(self, job) -> None:
        frozen, directory = read_request(self.projects.runtime, job)
        base = frozen_bytes(directory, "base.png", frozen["base_sha"])
        mask = frozen_bytes(directory, "mask.png", frozen["mask_sha"]) if frozen.get("mask_sha") is not None else None
        receipt = self._submit_frozen(frozen, base, mask)
        self._finish(
            job, local_status=Status.QUEUED, remote_status=receipt.status, remote_task_id=receipt.task_id,
            error_code=None, next_action_at=self._delay(job),
        )

    def _poll(self, job) -> None:
        task = self.provider.poll(job["remote_task_id"], self.kind)
        next_time = self._delay(job)
        if task.status in {"PENDING", "QUEUED", "RUNNING"}:
            self._finish(
                job, local_status=Status.RUNNING if task.status == "RUNNING" else Status.QUEUED,
                remote_status=task.status, poll_count=job["poll_count"] + 1,
                next_action_at=next_time, error_code=None,
            )
            return
        if task.status == "SUCCEEDED":
            expired = job["link_refreshes"] and (not task.url or task.url == job["result_url"])
            if expired:
                status, error = Status.RESULT_EXPIRED, "RESULT_EXPIRED"
            elif not task.url:
                status, error = Status.FAILED, "RESULT_MISSING"
            else:
                status, error = Status.REMOTE_SUCCEEDED, None
        elif task.status == "UNKNOWN":
            status, error = Status.RESULT_EXPIRED, "RESULT_EXPIRED"
        else:
            status, error = Status.FAILED, task.failure or "PROVIDER_REJECTED"
        with connect(self.projects.database) as db:
            changed = self._change(
                db, job, local_status=status, remote_status=task.status,
                result_url=task.url if status == Status.REMOTE_SUCCEEDED else None,
                next_action_at=self.clock(), poll_count=0, error_code=error,
            )
            if changed and task.status != "UNKNOWN":
                db.execute(
                    """INSERT INTO usage_events
                       (id, job_id, provider, model, image_count, video_seconds, resolution, audio, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(job_id) DO UPDATE SET
                       image_count = COALESCE(usage_events.image_count, excluded.image_count),
                       video_seconds = COALESCE(usage_events.video_seconds, excluded.video_seconds),
                       resolution = COALESCE(usage_events.resolution, excluded.resolution),
                       audio = COALESCE(usage_events.audio, excluded.audio)""",
                    (new_id(), job["id"], PROVIDER, job["model"], task.image_count, task.video_seconds,
                     task.resolution, task.audio, utc_now()),
                )

    def _record_result(self, job, source, output, frozen) -> None:
        with connect(self.projects.database) as db:
            db.execute("BEGIN IMMEDIATE")
            changed = self._change(
                db, job, local_status=Status.SUCCEEDED, result_path=output.relative_path,
                result_url=None, error_code=None, paused=0,
            )
            if changed:
                self.projects.record_generated(
                    db, source, output.id, job["id"], job["source_revision"], frozen,
                )

    def _handle_error(self, job, exc: AppError) -> None:
        status = job["local_status"]
        if status == Status.SUBMITTING:
            self._finish(
                job, local_status=Status.SUBMISSION_UNKNOWN if exc.code == "SUBMIT_UNKNOWN" else Status.FAILED,
                error_code=exc.code,
            )
        elif status == Status.DOWNLOADING:
            if exc.code == "RESULT_LINK_EXPIRED" and not job["link_refreshes"]:
                self._finish(job, local_status=Status.RUNNING, error_code=exc.code, link_refreshes=1,
                             next_action_at=self.clock(), poll_count=0)
            elif exc.retryable:
                self._finish(job, local_status=Status.REMOTE_SUCCEEDED, error_code=exc.code,
                             next_action_at=self._delay(job), poll_count=job["poll_count"] + 1)
            else:
                state = Status.RESULT_EXPIRED if exc.code == "RESULT_LINK_EXPIRED" else Status.FAILED
                self._finish(job, local_status=state, error_code=exc.code, result_url=None)
        elif exc.retryable:
            self._finish(job, error_code=exc.code, next_action_at=self._delay(job), poll_count=job["poll_count"] + 1)
        else:
            self._finish(job, error_code=exc.code, paused=1)


class StaticJobs(MediaJobs):
    kind = JobKind.IMAGE
    supported_model = "wanx2.1-imageedit"

    def submit(self, project_id: str, expected_id: str, instruction: str, local: bool, protect: bool = False) -> str:
        if not self.ready:
            raise AppError("PROVIDER_CONFIG", "请配置本机密钥及已验证的万相 2.1 媒体地址和模型。")
        source = self.projects.require_current(project_id, expected_id)
        if source.base_video_asset_id is not None:
            raise AppError("IMAGE_SOURCE", "请在历史中打开静态锚点，再提交静态生成。")
        request, base, mask = freeze_request(self.projects, source, self.model, instruction, local, protect)
        return self._queue(source, request, base, mask)

    def _submit_frozen(self, frozen, base, mask):
        return self.provider.submit(frozen, base, mask)

    def _download(self, job) -> None:
        frozen, _ = read_request(self.projects.runtime, job)
        source = self.projects.version(job["project_id"], job["source_version_id"])
        base_asset = self.projects.asset(source.base_image_asset_id)
        if base_asset.id != frozen["base_asset_id"] or base_asset.sha256 != frozen["base_sha256"]:
            raise AppError("JOB_INPUT", "请求底图与历史版本不一致。")
        base = self.projects.load_image(base_asset)
        mask = None
        if frozen["protect"]:
            region = self.projects.region(source)
            if region is None or region.id != frozen["region_id"] or region.mask_asset_id != frozen["mask_asset_id"]:
                raise AppError("JOB_INPUT", "保存的选区与发送时的版本不一致。")
            mask = self.projects.load_image(self.projects.asset(region.mask_asset_id))
        result = decode_result(self.provider.download(job["result_url"]))
        mapped = map_result(result, ImageGeometry(**frozen["geometry"]), base, mask)
        try:
            output = self.projects.put_image(mapped, AssetKind.IMAGE)
        except AppError as exc:
            raise AppError("RESULT_SAVE", "结果已生成，本机保存暂时失败，可继续下载。", retryable=True) from exc
        self._record_result(job, source, output, frozen)
