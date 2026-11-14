
"""Local SAM 2.1 point/box drafts bound to the version on which they were made."""

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .errors import AppError
from .image_files import on_white, png_bytes
from .media_io import file_sha256
from .model_manager import ModelSlot
from .regions import ViewTransform, tint_mask, view_image
from .schemas import Asset, Version


SAM_MODEL = "SAM 2.1 Hiera Tiny"
SAM_CHECKPOINT = "sam2.1_hiera_tiny.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_HASH = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
MAX_POINTS = 32


@dataclass(frozen=True)
class Candidate:
    mask: bytes
    score: float


@dataclass(frozen=True)
class SegmentationDraft:
    binding: tuple[str, str, str, str]
    points: tuple[tuple[int, int, int], ...] = ()
    corners: tuple[tuple[int, int], ...] = ()
    candidates: tuple[Candidate, ...] = ()


def binding(snapshot: Version, asset: Asset) -> tuple[str, str, str, str]:
    return snapshot.project_id, snapshot.id, asset.id, asset.sha256


def add_hint(draft: SegmentationDraft, transform: ViewTransform, point, hint: str) -> SegmentationDraft:
    x, y = transform.point(point)
    if hint == "box":
        corners = (*draft.corners, (x, y)) if len(draft.corners) == 1 else ((x, y),)
        if len(corners) == 2:
            transform.box(*corners)
        return replace(draft, corners=corners, candidates=())
    if hint not in {"positive", "negative"}:
        raise AppError("SAM_HINT", "请选择正点、负点或框提示。")
    points = tuple(item for item in draft.points if item[:2] != (x, y))
    if len(points) >= MAX_POINTS:
        raise AppError("SAM_POINTS", f"每次最多记录 {MAX_POINTS} 个点，或清空提示后重选。")
    return replace(draft, points=(*points, (x, y, int(hint == "positive"))), candidates=())


def preview_hints(clean: Image.Image, draft: SegmentationDraft) -> Image.Image:
    output = clean.convert("RGB")
    draw = ImageDraw.Draw(output)
    radius = max(2, min(7, max(clean.size) // 100))
    for x, y, label in draft.points:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="#08A551" if label else "#DA3035",
                     outline="white", width=2)
    if len(draft.corners) == 1:
        x, y = draft.corners[0]
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="#0577CD", outline="white", width=2)
    if len(draft.corners) == 2:
        (x0, y0), (x1, y1) = draft.corners
        draw.rectangle((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), outline="#0577CD", width=2)
    return output


def candidate_mask(candidate: Candidate, size: tuple[int, int]) -> Image.Image:
    with Image.open(BytesIO(candidate.mask), formats=["PNG"]) as image:
        if image.mode != "L" or image.size != size:
            raise AppError("SAM_MASK", "候选蒙版与当前底图不匹配。")
        counts = image.histogram()
        if counts[255] == 0 or counts[0] + counts[255] != image.width * image.height:
            raise AppError("SAM_MASK", "候选蒙版为空或格式无效。")
        return image.copy()


def candidate_preview(clean: Image.Image, candidate: Candidate) -> Image.Image:
    return tint_mask(clean, candidate_mask(candidate, clean.size))


def save_candidate(editor, snapshot: Version, draft: SegmentationDraft | None, index: int) -> Version:
    asset = editor.projects.asset(snapshot.base_image_asset_id)
    if draft is None or draft.binding != binding(snapshot, asset):
        raise AppError("SAM_STALE", "候选来自另一底图或版本，请重新生成。")
    if type(index) is not int or not 0 <= index < len(draft.candidates):
        raise AppError("SAM_CHOICE", "请先选择一个候选蒙版。")
    transform = ViewTransform.for_base((asset.width, asset.height))
    mask = candidate_mask(draft.candidates[index], transform.view_size)
    mask = mask.resize(transform.base_size, Image.Resampling.NEAREST)
    prompts = {
        "mode": "sam2", "model": SAM_MODEL, "checkpoint_sha256": SAM_HASH,
        "view_points": draft.points, "view_box_corners": draft.corners, "candidate": index + 1,
        "score": round(draft.candidates[index].score, 6),
    }
    if len(draft.corners) == 2:
        prompts["base_box"] = transform.box(*draft.corners)
    return editor._save_region(snapshot, asset, transform, mask, prompts)


class SamSegmenter:
    def __init__(self, model_directory: Path, slot: ModelSlot):
        self.checkpoint = model_directory / "sam2" / SAM_CHECKPOINT
        self.slot = slot
        self._predictor = None
        self._encoded = None
        self._verified = None
        self._prefer_cpu = False
        self.last_device = None

    def _release(self) -> None:
        self._predictor = self._encoded = None

    def _load(self, torch):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        stat = self.checkpoint.stat()
        stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if stamp != self._verified:
            if file_sha256(self.checkpoint) != SAM_HASH:
                raise AppError("SAM_WEIGHTS", "SAM 权重校验失败，请检查本机模型文件。")
            self._verified = stamp
        device = "cuda" if torch.cuda.is_available() and not self._prefer_cpu else "cpu"
        model = build_sam2(SAM_CONFIG, str(self.checkpoint), device=device, apply_postprocessing=False)
        self._predictor = SAM2ImagePredictor(model)
        self.last_device = device

    def _predict(self, clean, image_key, points, labels, box):
        import torch

        self.slot.claim(self, self._release)
        with torch.inference_mode():
            while True:
                try:
                    if self._predictor is None:
                        self._load(torch)
                    if self._encoded != image_key:
                        self._encoded = None
                        self._predictor.set_image(np.array(clean))
                        self._encoded = image_key
                    return self._predictor.predict(
                        point_coords=points, point_labels=labels, box=box, multimask_output=True,
                    )[:2]
                except torch.cuda.OutOfMemoryError:
                    if self._prefer_cpu:
                        raise
                self._release()
                self._prefer_cpu = True
                torch.cuda.empty_cache()

    def segment(self, clean: Image.Image, draft: SegmentationDraft) -> SegmentationDraft:
        if len(draft.corners) == 1:
            raise AppError("SAM_BOX", "请先点完框的第二角，或清空提示。")
        if not any(label for _, _, label in draft.points) and not draft.corners:
            raise AppError("SAM_HINT", "请先添加一个正点，或用两个对角框出主体。")
        points = np.array([(x, y) for x, y, _ in draft.points], dtype=np.float32) if draft.points else None
        labels = np.array([label for _, _, label in draft.points], dtype=np.int32) if draft.points else None
        box = None
        if draft.corners:
            (x0, y0), (x1, y1) = draft.corners
            box = np.array([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], dtype=np.float32)
        try:
            project, _, asset, digest = draft.binding
            with self.slot.lock:
                masks, scores = self._predict(on_white(clean), (project, asset, digest, clean.size), points, labels, box)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise AppError("SAM_INFERENCE", "本机 SAM 推理失败。请检查模型环境，或继续使用框选和画笔。") from exc
        if masks.ndim != 3 or masks.shape[1:] != (clean.height, clean.width) or scores.shape != (len(masks),):
            raise AppError("SAM_MASK", "SAM 候选尺寸无效，请改用框选或画笔。")
        candidates = tuple(
            Candidate(png_bytes(Image.fromarray(masks[i].astype(bool).astype(np.uint8) * 255)), float(scores[i]))
            for i in np.argsort(-scores) if np.isfinite(scores[i]) and masks[i].any()
        )[:3]
        if not candidates:
            raise AppError("SAM_MASK", "没有得到非空候选。请调整提示点或框。")
        return replace(draft, candidates=candidates)
