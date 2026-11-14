
"""Turn a user's instruction into an editable draft. Plans never execute actions."""

import json
import logging
import re
from dataclasses import dataclass, replace

import httpx

from .constraints import constraint_fields, removal_target
from .errors import AppError
from .schemas import IntentAction as Action, IntentPlan


TEXT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_INPUT = 600
MAX_RESPONSE = 24_576
TEXT_LIMITS = {
    "search_query": 150, "image_instruction": 600, "video_instruction": 600,
    "caption": 240, "clarification_text": 240,
}
BOOL_FIELDS = {"region_required", "needs_clarification"}
REQUIRED = {"action", "constraint_add", "constraint_remove", *TEXT_LIMITS, *BOOL_FIELDS}
CONSTRAINT_FIELDS = {"category", "target", "text", "scope", "duration"}
LOGGER = logging.getLogger(__name__)
_SPLIT = re.compile(
    r"\s*(?:[，,；;。]\s*(?:然后|再|接着|之后|并且)?|(?:然后|之后|接着|再(?=给)|并(?=给)))\s*"
    r"(?=(?:给|把|将|让|做成|改成|换成|戴上|加上|加一|添加|去掉|删除|生成))"
)
_SEARCH = re.compile(r"^(?:请)?(?:帮我)?(?:在(?:本地)?图库(?:里)?|从(?:本地)?图库(?:里)?)?(?:找(?:到|出)?|搜(?:索|一下)?|检索)(?:一张|一幅|一只|一幅图)?\s*")
_CAPTION = re.compile(r"^(?:请)?(?:把)?(?:配字|字幕|文案)(?:改成|换成|设为|写成|：|:)\s*(.+)$")
_EDIT = re.compile(r"^(?:请)?(?:给|把|将|让|去掉|加上|添加|换成|改成|只改|局部)")


@dataclass(frozen=True)
class IntentContext:
    page: str
    selected: bool
    version_kind: str = ""
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentDraft:
    plan: IntentPlan
    note: str


def _clarify(message: str) -> IntentPlan:
    return IntentPlan(Action.CLARIFY, needs_clarification=True, clarification_text=message)


def _clean(text: str | None) -> str:
    value = (text or "").strip()
    if len(value) > MAX_INPUT:
        raise AppError("INTENT_LENGTH", f"指令超过 {MAX_INPUT} 字，请缩短或手动填写检索和修改说明。")
    if not value:
        raise AppError("INTENT_EMPTY", "请先填写一条指令。")
    return value


def local_plan(text: str, context: IntentContext) -> IntentPlan:
    text = _clean(text)
    short = text.strip("。！! ")
    if short in {"撤销", "撤销一步", "回到上一步"}:
        return IntentPlan(Action.UNDO)
    if short in {"导出", "导出图片", "导出 PNG", "导出PNG"}:
        return IntentPlan(Action.EXPORT)
    if short.startswith(("恢复版本", "恢复历史版本")):
        return IntentPlan(Action.CHECKOUT)
    target = removal_target(text)
    if target and len(target) <= 32:
        return IntentPlan(Action.UPDATE_CONSTRAINTS, constraint_remove=(target,))
    caption = _CAPTION.fullmatch(text)
    if caption:
        if len(caption[1]) > TEXT_LIMITS["caption"]:
            return _clarify("文案较长，请在本地配字页手动填写。")
        return IntentPlan(Action.EDIT_TEXT, caption=caption[1].strip("“”「」"))
    prefix = _SEARCH.match(text)
    if prefix:
        parts = _SPLIT.split(text[prefix.end():], maxsplit=1)
        search = parts[0].strip(" ，。；")
        if not search or len(search) > TEXT_LIMITS["search_query"]:
            return _clarify("请填写不超过 150 字的检索描述。")
        if len(parts) == 2 and parts[1].strip():
            return IntentPlan(Action.SEARCH_THEN_EDIT, search_query=search, image_instruction=parts[1].strip())
        return IntentPlan(Action.SEARCH, search_query=search)
    if context.selected and (context.page == "static" or _EDIT.match(text)):
        return IntentPlan(Action.EDIT_IMAGE, image_instruction=text, region_required=text.startswith(("只改", "局部")))
    return _clarify("请手动填写检索描述或编辑要求，也可以明确写「找一张…，给它…」。")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def validate_plan(data: str) -> IntentPlan:
    if type(data) is not str or not 1 <= len(data) <= 12_000:
        raise ValueError("invalid reply length")
    payload = json.loads(data, object_pairs_hook=_unique_object)
    if not isinstance(payload, dict) or payload.keys() != REQUIRED:
        raise ValueError("missing or unknown fields")
    action = Action(payload["action"])
    fields = {}
    for field, limit in TEXT_LIMITS.items():
        value = payload[field]
        if type(value) is not str or len(value) > limit:
            raise ValueError("invalid text field")
        fields[field] = value.strip()
    for field in BOOL_FIELDS:
        if type(payload[field]) is not bool:
            raise ValueError("invalid flag")
        fields[field] = payload[field]
    additions, removals = payload["constraint_add"], payload["constraint_remove"]
    if not isinstance(additions, list) or not isinstance(removals, list) or max(len(additions), len(removals)) > 8:
        raise ValueError("invalid constraint list")
    if not all(isinstance(item, dict) and item.keys() == CONSTRAINT_FIELDS for item in additions):
        raise ValueError("invalid constraint entry")
    if not all(type(item) is str and 1 <= len(item.strip()) <= 32 for item in removals):
        raise ValueError("invalid constraint removal")
    additions = tuple(constraint_fields(**item) for item in additions)
    must_have = {
        Action.SEARCH: ("search_query",), Action.SEARCH_THEN_EDIT: ("search_query", "image_instruction"),
        Action.EDIT_IMAGE: ("image_instruction",), Action.EDIT_TEXT: ("caption",),
        Action.GENERATE_VIDEO: ("video_instruction",), Action.EDIT_VIDEO_INSTRUCTION: ("video_instruction",),
    }
    if any(not fields[field] for field in must_have.get(action, ())):
        raise ValueError("incomplete action")
    if action == Action.UPDATE_CONSTRAINTS and not (additions or removals):
        raise ValueError("missing constraint changes")
    if action == Action.CLARIFY and not fields["needs_clarification"]:
        raise ValueError("invalid clarification")
    if fields["needs_clarification"] and not fields["clarification_text"]:
        raise ValueError("missing clarification")
    return IntentPlan(action, **fields, constraint_add=additions, constraint_remove=tuple(item.strip() for item in removals))


SYSTEM_PROMPT = (
    "你为个人图库表情包工具拆解指令。只输出 JSON 对象，不执行操作或访问图片。"
    "字段必须齐全，除这些字段外不要输出任何字段。"
    f" action 为 {', '.join(Action)} 之一。"
    " search_query、image_instruction、video_instruction、caption、clarification_text 为字符串，缺省用空字符串。"
    " region_required、needs_clarification 为布尔值。constraint_add、constraint_remove 为数组，缺省为空数组。"
    " constraint_remove 元素为要移除的目标名称。constraint_add 元素是带 category、target、text、scope、duration 的对象。"
    " category 选 subject/region/style/motion/text，scope 选 image/video/media/caption，duration 选 persistent/once。"
    " text 类仅用于 caption，motion 类仅用于 video。"
    " 区分寻找已有画面和先检索再改图。『找戴墨镜的橘猫』是 search，描述为『戴墨镜的橘猫』。"
    " 输入『找一张橘猫，给它加墨镜』时，完整输出示例为："
    '{"action":"search_then_edit","search_query":"橘猫","image_instruction":"给它加墨镜",'
    '"video_instruction":"","caption":"","region_required":false,"constraint_add":[],'
    '"constraint_remove":[],"needs_clarification":false,"clarification_text":""}'
    " 其他输入也须返回示例中的全部字段，未使用的布尔字段填 false。"
    " 仅改配字时不要填写画面修改。删除保留要求写入 constraint_remove。"
    " 有矛盾或信息不足时用 clarify，needs_clarification 为 true，并填写 clarification_text。"
)


class IntentParser:
    def __init__(self, base_url: str, model: str, key: str, *, transport=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self._key = key
        self._transport = transport

    @property
    def ready(self) -> bool:
        return bool(self._key and self.model and self.url == TEXT_ENDPOINT + "/chat/completions")

    def draft(self, raw: str | None, context: IntentContext, *, cloud: bool = False) -> IntentDraft:
        try:
            text = _clean(raw)
        except AppError as exc:
            return IntentDraft(_clarify(exc.user_message), exc.user_message)
        local = local_plan(text, context)
        if not cloud or local.action in {Action.UNDO, Action.EXPORT, Action.CHECKOUT, Action.EDIT_TEXT}:
            return IntentDraft(local, "本机拆解完成。" if not local.needs_clarification else local.clarification_text)
        if not self.ready:
            return IntentDraft(local, "文字服务未配置，已使用本机拆解。")
        try:
            plan = self._request(text, context)
        except (httpx.HTTPError, AppError) as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f"HTTP {exc.response.status_code}"
            else:
                detail = exc.code if isinstance(exc, AppError) else type(exc).__name__
            LOGGER.info("文字拆解使用本机回退，原因 %s", detail)
            return IntentDraft(local, "云端未返回可用的拆解，已回退到本机。请核对后再操作。")
        if plan.needs_clarification:
            plan = replace(plan, action=Action.CLARIFY)
        return IntentDraft(plan, "云端拆解完成，请核对并按需修改。")

    def _request(self, text: str, context: IntentContext) -> IntentPlan:
        user = {
            "page": context.page, "selected": context.selected, "version_kind": context.version_kind,
            "constraints": context.constraints, "instruction": text,
        }
        request = {
            "model": self.model, "enable_thinking": False, "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        }
        data = bytearray()
        with httpx.Client(timeout=20, follow_redirects=False, transport=self._transport) as client:
            with client.stream("POST", self.url, json=request, headers={"Authorization": f"Bearer {self._key}"}) as reply:
                reply.raise_for_status()
                for chunk in reply.iter_bytes():
                    if len(data) + len(chunk) > MAX_RESPONSE:
                        raise AppError("INTENT_RESPONSE_SIZE", "文字服务响应超过长度上限。")
                    data.extend(chunk)
        try:
            choice = json.loads(data)["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, TypeError, IndexError, RecursionError) as exc:
            raise AppError("INTENT_ENVELOPE", "文字服务响应格式无法识别。") from exc
        if choice.get("finish_reason") != "stop":
            raise AppError("INTENT_FINISH", "文字服务未完整返回拆解。")
        try:
            return validate_plan(content)
        except (AppError, ValueError, TypeError, RecursionError) as exc:
            raise AppError("INTENT_SCHEMA", "文字服务返回的拆解字段校验失败。") from exc


ACTION_LABELS = {
    Action.SEARCH: "检索已有图片", Action.SEARCH_THEN_EDIT: "先检索，再修改所选图片",
    Action.EDIT_IMAGE: "修改画面", Action.EDIT_TEXT: "本地配字",
    Action.GENERATE_VIDEO: "生成视频（P6）", Action.EDIT_VIDEO_INSTRUCTION: "修改视频要求（P6）",
    Action.UPDATE_CONSTRAINTS: "调整保留要求", Action.UNDO: "撤销一步",
    Action.CHECKOUT: "恢复历史版本", Action.EXPORT: "导出", Action.CLARIFY: "待确认",
}


def describe_draft(draft: IntentDraft) -> str:
    plan = draft.plan
    lines = [draft.note, f"建议操作：{ACTION_LABELS[plan.action]}。"]
    if plan.needs_clarification and plan.clarification_text != draft.note:
        lines.append(plan.clarification_text)
    if plan.region_required and not plan.needs_clarification:
        lines.append("修改前请确认是否需要局部选区。")
    if plan.caption:
        lines.append(f"配字建议：{plan.caption}")
    if plan.video_instruction:
        lines.append(f"视频要求：{plan.video_instruction}")
    for item in plan.constraint_add:
        lines.append(f"待确认添加：{item['target']} · {item['text']}")
    if plan.constraint_remove:
        lines.append("待确认移除：" + "、".join(plan.constraint_remove))
    return "\n".join(lines)
