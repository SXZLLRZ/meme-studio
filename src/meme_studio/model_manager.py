"""Lazy local Chinese-CLIP inference with serialized access to the GPU."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image

from .errors import AppError
from .image_files import on_white
from .media_io import file_sha256


CHECKPOINT_NAME = "clip_cn_vit-b-16.pt"
CHECKPOINT_SHA256 = "5997ef32c50de18f3ac06eb5d746c324a639053c3bed8f2986f97ebb22b2ecde"
BATCH_SIZE = 4


@dataclass(frozen=True)
class ModelSignature:
    model_name: str
    checkpoint_hash: str
    preprocess_version: str

    def key(self) -> tuple[str, str, str]:
        return (self.model_name, self.checkpoint_hash, self.preprocess_version)


def unit_rows(vectors) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.size:
        raise AppError("VECTOR_SHAPE", "模型返回的向量维度无效。")
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norm).all() or np.any(norm <= 0):
        raise AppError("VECTOR_VALUE", "模型返回了无效的向量。")
    return np.ascontiguousarray(matrix / norm, dtype=np.float32)


class ModelSlot:
    """A shared inference lock with one resident model at a time."""

    def __init__(self):
        self.lock = RLock()
        self._owner = None
        self._release = None

    def claim(self, owner, release) -> None:
        if self._owner is owner:
            return
        if self._release is not None:
            self._release()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._owner, self._release = owner, release


class ModelManager:
    def __init__(self, model_directory: Path, slot: ModelSlot | None = None):
        self.checkpoint = model_directory / CHECKPOINT_NAME
        self.slot = slot if slot is not None else ModelSlot()
        self.gpu_lock = self.slot.lock
        self._verified: tuple[int, int, int] | None = None
        self._loaded = None
        self._prefer_cpu = False
        self._last_device: str | None = None

    @property
    def last_device(self) -> str | None:
        return self._last_device

    def signature(self) -> ModelSignature:
        with self.gpu_lock:
            try:
                clip_version = version("cn-clip")
                vision_version = version("torchvision")
                pillow_version = version("Pillow")
                version("torch")
            except PackageNotFoundError as exc:
                raise AppError("MODEL_DEPENDENCY", "语义检索未就绪：缺少本地 Chinese-CLIP 推理依赖。") from exc
            try:
                stat = self.checkpoint.stat()
                stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if self._verified != stamp:
                    if file_sha256(self.checkpoint) != CHECKPOINT_SHA256:
                        raise AppError("MODEL_HASH", "语义检索未就绪：模型权重校验失败。")
                    self._verified = stamp
                    self._loaded = None
            except OSError as exc:
                raise AppError("MODEL_WEIGHTS", "语义检索未就绪：未找到已校验的本地模型权重。") from exc
            return ModelSignature(
                "Chinese-CLIP ViT-B-16", CHECKPOINT_SHA256,
                f"cn-clip-{clip_version}/torchvision-{vision_version}/pillow-{pillow_version}/srgb-white-v1",
            )

    def _load(self, torch):
        if self._loaded is None:
            from cn_clip.clip import load_from_name, tokenize

            device = "cuda" if torch.cuda.is_available() and not self._prefer_cpu else "cpu"
            model, preprocess = load_from_name(
                str(self.checkpoint),
                device=device,
                vision_model_name="ViT-B-16",
                text_model_name="RoBERTa-wwm-ext-base-chinese",
                input_resolution=224,
            )
            self._loaded = (model.eval(), preprocess, tokenize, device)
        return self._loaded

    def _release(self) -> None:
        self._loaded = None

    def _run(self, operation) -> np.ndarray:
        with self.gpu_lock:
            self.signature()
            try:
                import torch

                self.slot.claim(self, self._release)
                while True:
                    try:
                        with torch.inference_mode():
                            output = operation(torch, self._load(torch))
                        self._last_device = self._loaded[3]
                        return unit_rows(output)
                    except torch.cuda.OutOfMemoryError:
                        if self._prefer_cpu:
                            raise
                    self._loaded = None
                    self._prefer_cpu = True
                    torch.cuda.empty_cache()
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise AppError("MODEL_INFERENCE", "本地 Chinese-CLIP 推理失败，可继续浏览和手动选图。") from exc

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        def predict(torch, loaded):
            model, preprocess, _, device = loaded
            tensors = []
            for path in paths:
                with Image.open(path, formats=["PNG"]) as image:
                    tensors.append(preprocess(on_white(image)))
            try:
                return model.encode_image(torch.stack(tensors).to(device)).float().cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                if len(tensors) == 1:
                    raise
            return np.concatenate([
                model.encode_image(tensor.unsqueeze(0).to(device)).float().cpu().numpy()
                for tensor in tensors
            ])

        return self._run(predict)

    def encode_text(self, query: str) -> np.ndarray:
        def predict(torch, loaded):
            model, _, tokenize, device = loaded
            return model.encode_text(tokenize([query]).to(device)).float().cpu().numpy()

        return self._run(predict)
