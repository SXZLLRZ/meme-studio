
"""Versioned requirements and deterministic checks for explicit target changes."""

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, replace

from .errors import AppError
from .schemas import Version, new_id


CATEGORIES = {"subject": "主体特征", "region": "画面区域", "style": "风格", "motion": "动作", "text": "配字"}
SCOPES = {"image": "静态画面", "video": "动态画面", "media": "静态和动态画面", "caption": "本地配字"}
DURATIONS = {"persistent": "持续", "once": "仅下一次对应操作"}
ALIASES = {"背景": ("背景", "背景图", "底色"), "条纹": ("条纹", "花纹"), "毛色": ("毛色", "皮毛颜色")}
_KEEP = r"不(?:要|许)?(?:更换|改变|改动|修改|换|动)|(?:保持|保留|维持)(?!下述)|不变|原样"
_CHANGE = r"(?<!不)(?<!不要)(?<!不许)(?:替换(?:成|为)?|换(?:成|为|掉)|改(?:成|为|掉)|变(?:成|为)|替换|更换|去掉|删除|移除|调成)"
_REMOVE = re.compile(r"^(?:请)?(?:不要再|不再|无需再|不用再)(?:保留|保持|保护)\s*(.+)$")


def _target(text: str) -> str:
    key = re.sub(r"[\s，。！!？?]+", "", unicodedata.normalize("NFKC", text)).casefold()
    return next((name for name, aliases in ALIASES.items() if key in aliases), key)


@dataclass(frozen=True)
class Constraint:
    id: str
    category: str
    target: str
    text: str
    scope: str
    duration: str
    source_version_id: str

    def applies(self, stage: str) -> bool:
        return self.scope == stage or self.scope == "media" and stage in {"image", "video"}


def constraint_fields(category, target, text, scope, duration) -> dict[str, str]:
    if not all(type(value) is str for value in (category, target, text, scope, duration)):
        raise AppError("CONSTRAINT_FIELDS", "请填写目标和保留要求。")
    target, text = target.strip(), text.strip()
    if (
        category not in CATEGORIES or scope not in SCOPES or duration not in DURATIONS
        or (category == "text") != (scope == "caption") or category == "motion" and scope != "video"
    ):
        raise AppError("CONSTRAINT_SCOPE", "配字要求请选择本地配字，动作要求请选择动态画面。")
    if not _target(target) or len(target) > 32 or not text or len(text) > 160:
        raise AppError("CONSTRAINT_FIELDS", "目标填 1 至 32 字，具体要求填 1 至 160 字。")
    return dict(category=category, target=target, text=text, scope=scope, duration=duration)


def read_constraints(snapshot: Version) -> tuple[Constraint, ...]:
    entries = json.loads(snapshot.constraints_json).get("constraints", [])
    result = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("active", True):
            continue
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        scope = {"motion": "video", "text": "caption", "both": "media"}.get(entry.get("scope"), entry.get("scope", "image"))
        if scope not in SCOPES:
            scope = "image"
        category = entry.get("category", {"caption": "text", "video": "motion"}.get(scope, "subject"))
        result.append(Constraint(
            entry.get("id", f"legacy:{index}"), category, entry.get("target", text), text.strip(), scope,
            entry.get("duration", "persistent"), entry.get("source_version_id", snapshot.id),
        ))
    return tuple(result)


def constraint_document(entries: tuple[Constraint, ...]) -> str:
    return json.dumps({"schema_version": 1, "constraints": [asdict(entry) for entry in entries]},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def constraint_summary(snapshot: Version) -> str:
    return "\n".join(
        f"{entry.target} · {SCOPES[entry.scope]} · {DURATIONS.get(entry.duration, '持续')} · "
        f"来源 {entry.source_version_id[:8]}：{entry.text}"
        for entry in read_constraints(snapshot)
    ) or "此版本暂无保留要求。"


def save_constraint(snapshot: Version, selected: str | None, **fields) -> tuple[str, tuple[str, ...]]:
    fields = constraint_fields(**fields)
    entries = read_constraints(snapshot)
    target = _target(fields["target"])
    current = next((entry for entry in entries if entry.id == selected), None)
    if selected and current is None:
        raise AppError("CONSTRAINT_SELECTION", "这项要求不在当前版本中，请重新选择。")
    other = next((entry for entry in entries if entry.id != selected and _target(entry.target) == target), None)
    replacement = Constraint(selected or (other.id if other else new_id()), **fields, source_version_id=snapshot.id)
    if current and not other and replace(replacement, source_version_id=current.source_version_id) == current:
        return snapshot.constraints_json, ()
    removed = {selected, other.id if other else None}
    result = tuple(entry for entry in entries if entry.id not in removed) + (replacement,)
    if len(result) > 24:
        raise AppError("CONSTRAINT_LIMIT", "一个版本最多保存 24 项要求，请先移除不用的要求。")
    validate_requirements(result)
    return constraint_document(result), (other.text,) if other else ()


def delete_constraint(snapshot: Version, selected: str | None) -> str:
    entries = read_constraints(snapshot)
    result = tuple(entry for entry in entries if entry.id != selected)
    if len(result) == len(entries):
        raise AppError("CONSTRAINT_SELECTION", "请先选择当前版本的一项要求。")
    return constraint_document(result)


def removal_target(instruction: str) -> str | None:
    match = _REMOVE.fullmatch(instruction.strip(" \t。！"))
    return match[1].strip() if match else None


def _target_flags(text: str, target: str) -> tuple[bool, bool]:
    names = ALIASES.get(target, (target,))
    if not target or len(target) > 32:
        return False, False
    name = "(?:" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + ")"
    keep = change = previous_target = False
    clauses = re.split(r"[\n，,。；;！？]|(?:并且|同时|但是|但要|然后|并|且|但)(?=(?:请|把|将|改|换|保持|保留|不))", text)
    for raw in clauses:
        clause = raw.strip()
        if previous_target and re.match(r"(?:换|改|替换|更换)(?:成|为|掉)|(?:保持|保留)(?:不变|原样)", clause):
            keep |= bool(re.match(_KEEP, clause))
            change |= bool(re.match(_CHANGE, clause))
        previous_target = bool(re.search(name, clause, re.IGNORECASE))
        if previous_target:
            keep |= bool(re.search(rf"{name}.{{0,10}}(?:{_KEEP})|(?:{_KEEP}).{{0,6}}{name}", clause, re.IGNORECASE))
            change |= bool(re.search(rf"{name}.{{0,10}}(?:{_CHANGE})|(?:{_CHANGE}).{{0,6}}{name}", clause, re.IGNORECASE))
    return keep, change


def _check_flags(target: str, flags: tuple[bool, bool]) -> None:
    if all(flags):
        raise AppError("CONSTRAINT_CONFLICT", f"「{target}」同时被要求保持原样和修改。请明确选一项后再继续。")


def validate_requirements(entries: tuple[Constraint, ...]) -> None:
    targets = {_target(entry.target) for entry in entries}
    for stage in ("image", "video", "caption"):
        for target in targets:
            flags = [_target_flags(entry.text, target) for entry in entries if entry.applies(stage)]
            _check_flags(target, (any(keep for keep, _ in flags), any(change for _, change in flags)))


def remaining_after(entries: tuple[Constraint, ...], stage: str) -> tuple[Constraint, ...]:
    return tuple(entry for entry in entries if entry.duration != "once" or not entry.applies(stage))


@dataclass(frozen=True)
class ImageRequirements:
    applied: tuple[Constraint, ...]
    replaced: tuple[Constraint, ...]
    remaining: tuple[Constraint, ...]


def image_requirements(snapshot: Version, instruction: str) -> ImageRequirements:
    entries = read_constraints(snapshot)
    if any(removal_target(part) for part in re.split(r"[\n，,。；;]", instruction)):
        raise AppError("CONSTRAINT_REMOVE", "请在「保留要求」中移除对应条目，再提交画面修改。")
    targets = set(ALIASES) | {_target(entry.target) for entry in entries}
    flags = {target: _target_flags(instruction, target) for target in targets}
    for target, actions in sorted(flags.items()):
        _check_flags(target, actions)
    replaced = tuple(entry for entry in entries if entry.applies("image") and flags[_target(entry.target)][1])
    applied = tuple(entry for entry in entries if entry.applies("image") and entry not in replaced)
    for target, (keep, change) in sorted(flags.items()):
        from_requirements = [_target_flags(entry.text, target) for entry in applied]
        _check_flags(target, (keep or any(k for k, _ in from_requirements),
                              change or any(c for _, c in from_requirements)))
    remaining = tuple(entry for entry in entries if entry not in replaced)
    # A static replacement leaves the video-only part of a shared requirement intact.
    remaining += tuple(replace(entry, scope="video") for entry in replaced if entry.scope == "media")
    return ImageRequirements(applied, replaced, remaining_after(remaining, "image"))
