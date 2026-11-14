"""DashScope media adapter. No automatic POST retries."""

import base64
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from .errors import AppError
from .provider_image import MAX_INPUT_BYTES, MAX_RESULT_BYTES
from .schemas import JobKind


PROVIDER = "dashscope"
MEDIA_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1"
IMAGE_PATH = "/services/aigc/image2image/image-synthesis"
VIDEO_PATH = "/services/aigc/video-generation/video-synthesis"
MAX_VIDEO_BYTES = 96 * 1024**2
VIDEO_RESULT_HOST = re.compile(r"dashscope-[a-z0-9-]+\.(?:oss-accelerate|oss-cn-[a-z0-9-]+)\.aliyuncs\.com")
POLL_STATES = {"PENDING", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}
TASK_ID = re.compile(r"[A-Za-z0-9_-]{1,160}")
RESULT_HOST = re.compile(r"dashscope-(?:result-[a-z0-9-]+|[0-9]+)\.oss-cn-[a-z0-9-]+\.aliyuncs\.com")


@dataclass(frozen=True)
class Receipt:
    task_id: str
    status: str | None


@dataclass(frozen=True)
class RemoteTask:
    status: str
    url: str | None = None
    image_count: int | None = None
    failure: str | None = None
    video_seconds: float | None = None
    resolution: str | None = None
    audio: bool | None = None


def _inline(data: bytes) -> str:
    if not 0 < len(data) <= MAX_INPUT_BYTES or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AppError("PROVIDER_INPUT", "发送图片副本格式或大小无效。")
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _reject(status: int, code: str = "") -> AppError:
    normalized = code.lower().replace("_", "")
    if status == 402 or any(term in normalized for term in ("balance", "arrear", "insufficient", "overdue")):
        return AppError("BALANCE", "平台余额或可用额度不足，请核对账户。")
    if status in {401, 403}:
        return AppError("AUTH", "平台拒绝鉴权或模型权限，请核对密钥和模型开通状态。")
    if status == 429:
        return AppError("RATE_LIMIT", "平台限流，请稍后再试。", retryable=True)
    if status >= 500:
        return AppError("PROVIDER_UNAVAILABLE", "平台暂时不可用。", retryable=True)
    return AppError("PROVIDER_REJECTED", "平台拒绝了当前请求，请核对模型、提示和图片参数。")


def _code(body) -> str:
    code = body.get("code") if isinstance(body, dict) else None
    return code if isinstance(code, str) else ""


def validate_result_url(url: str, *, video: bool = False) -> None:
    try:
        parsed = urlsplit(url)
        allowed = (
            parsed.scheme == "https" and parsed.port in {None, 443} and not parsed.username
            and not parsed.password and not parsed.fragment
            and (RESULT_HOST.fullmatch(parsed.hostname or "") or video and VIDEO_RESULT_HOST.fullmatch(parsed.hostname or ""))
        )
    except ValueError:
        allowed = False
    if not allowed:
        raise AppError("RESULT_URL", "平台返回的下载地址不在允许的 HTTPS 媒体域名范围内。")


class DashScope:
    def __init__(self, endpoint: str, key: str, client: httpx.Client | None = None):
        self.endpoint = endpoint.rstrip("/")
        self._key = key
        self._client = client if client is not None else httpx.Client(
            timeout=httpx.Timeout(25, connect=8), follow_redirects=False, trust_env=False,
        )
        self._owns_client = client is None

    @property
    def ready(self) -> bool:
        return bool(self._key) and self.endpoint == MEDIA_ENDPOINT

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _headers(self) -> dict:
        if not self.ready:
            raise AppError("PROVIDER_CONFIG", "请检查本机 API Key 和已验证的媒体地址。")
        return {"Authorization": f"Bearer {self._key}"}

    def submit(self, frozen: dict, base: bytes, mask: bytes | None) -> Receipt:
        content = {
            "function": frozen["function"], "prompt": frozen["prompt"],
            "base_image_url": _inline(base),
        }
        if mask is not None:
            content["mask_image_url"] = _inline(mask)
        return self._submit(IMAGE_PATH, frozen, content)

    def submit_video(self, frozen: dict, base: bytes) -> Receipt:
        return self._submit(VIDEO_PATH, frozen, {"prompt": frozen["prompt"], "img_url": _inline(base)})

    def _submit(self, path: str, frozen: dict, content: dict) -> Receipt:
        headers = {**self._headers(), "X-DashScope-Async": "enable"}
        payload = {"model": frozen["model"], "input": content, "parameters": frozen["parameters"]}
        try:
            response = self._client.post(self.endpoint + path, headers=headers, json=payload, follow_redirects=False)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise AppError("CONNECT", "未连接到平台，请检查网络后手动提交。") from exc
        except httpx.RequestError as exc:
            raise AppError("SUBMIT_UNKNOWN", "提交响应未完整收到，请核对平台控制台。") from exc

        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 500 or response.status_code in {408, 409}:
            raise AppError("SUBMIT_UNKNOWN", "平台响应无法确认是否创建任务，请核对平台控制台。")
        if not response.is_success or _code(body):
            raise _reject(response.status_code, _code(body))
        output = body.get("output") if isinstance(body, dict) else None
        task = output.get("task_id") if isinstance(output, dict) else None
        if not isinstance(task, str) or not TASK_ID.fullmatch(task):
            raise AppError("SUBMIT_UNKNOWN", "平台响应缺少可用任务 ID，请核对平台控制台。")
        status = output.get("task_status")
        return Receipt(task, status if isinstance(status, str) and status in POLL_STATES else None)

    def poll(self, task_id: str, kind: JobKind = JobKind.IMAGE) -> RemoteTask:
        if not TASK_ID.fullmatch(task_id):
            raise AppError("TASK_ID", "本地保存的任务 ID 无效。")
        try:
            response = self._client.get(self.endpoint + "/tasks/" + task_id, headers=self._headers(), follow_redirects=False)
        except httpx.RequestError as exc:
            raise AppError("QUERY_NETWORK", "查询暂时断开，将继续查询原任务。", retryable=True) from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code == 404 or _code(body) in {"TaskNotExist", "TaskNotFound", "TaskExpired"}:
            return RemoteTask("UNKNOWN")
        if not response.is_success or _code(body):
            raise _reject(response.status_code, _code(body))
        output = body.get("output") if isinstance(body, dict) else None
        status = output.get("task_status") if isinstance(output, dict) else None
        returned_id = output.get("task_id") if isinstance(output, dict) else None
        if not isinstance(status, str) or status not in POLL_STATES or (returned_id is not None and returned_id != task_id):
            raise AppError("QUERY_RESPONSE", "平台查询响应不完整，仍保留原任务 ID。", retryable=True)
        usage = body.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        if kind == JobKind.VIDEO:
            duration = usage.get("duration", usage.get("video_duration"))
            seconds = float(duration) if type(duration) in {int, float} and 0 < duration <= 30 else None
            resolution = usage.get("resolution")
            audio = usage.get("audio")
            url = output.get("video_url")
            return RemoteTask(
                status, url if status == "SUCCEEDED" and isinstance(url, str) and url else None,
                failure=_reject(400, _code(output)).code if status == "FAILED" else None,
                video_seconds=seconds,
                resolution=resolution if resolution in ("480P", "720P", "1080P") else None,
                audio=audio if type(audio) is bool else None,
            )
        count = usage.get("image_count")
        count = count if type(count) is int and count >= 0 else None
        if status != "SUCCEEDED":
            failure = _reject(400, _code(output)).code if status == "FAILED" else None
            return RemoteTask(status, image_count=count, failure=failure)
        results = output.get("results")
        url = next((item["url"] for item in results if isinstance(item, dict)
                    and isinstance(item.get("url"), str) and item["url"]), None) if isinstance(results, list) else None
        return RemoteTask(status, url, count)

    def download(self, url: str) -> bytes:
        return self._download(url, video=False)

    def download_video(self, url: str) -> bytes:
        return self._download(url, video=True)

    def _download(self, url: str, *, video: bool) -> bytes:
        common_types = {"", "application/octet-stream", "binary/octet-stream"}
        types = {"video/mp4"} if video else {"image/png", "image/jpeg", "image/jpg", "image/webp"}
        limit = MAX_VIDEO_BYTES if video else MAX_RESULT_BYTES
        try:
            for _ in range(3):
                validate_result_url(url, video=video)
                with self._client.stream("GET", url, follow_redirects=False) as response:
                    if response.is_redirect and response.headers.get("Location"):
                        url = urljoin(url, response.headers["Location"])
                        continue
                    if response.status_code in {401, 403, 404, 410}:
                        raise AppError("RESULT_LINK_EXPIRED", "结果地址已失效，尝试查询原任务获取链接。")
                    if not response.is_success:
                        raise AppError("DOWNLOAD_HTTP", "暂时无法下载，稍后会重试原结果。", retryable=True)
                    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                    if content_type not in common_types | types:
                        raise AppError("RESULT_TYPE", "结果链接返回的媒体类型不符合预期。", retryable=True)
                    length = response.headers.get("content-length", "")
                    if length.isascii() and length.isdigit() and len(length) < 20 and int(length) > limit:
                        raise AppError("RESULT_SIZE", "结果超过本地下载上限。")
                    data = bytearray()
                    for chunk in response.iter_bytes(64 * 1024):
                        data.extend(chunk)
                        if len(data) > limit:
                            raise AppError("RESULT_SIZE", "结果超过本地下载上限。")
                    return bytes(data)
        except httpx.RequestError as exc:
            raise AppError("DOWNLOAD_NETWORK", "下载中断，稍后会重试原结果。", retryable=True) from exc
        raise AppError("RESULT_URL", "结果地址重定向次数超出限制。")
