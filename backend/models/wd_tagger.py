"""wd-tagger v3 served through ONNX Runtime.

The model produces Danbooru-style tag confidences:
    0 = general
    4 = character
    9 = rating

Only categories 0 and 4 are persisted for search boost and tag browsing.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


log = logging.getLogger(__name__)

_CATEGORY_NAMES = {0: "general", 4: "character", 9: "rating"}

_CUDA12_PACKAGES = (
    # Order matters: load nvJitLink and NVRTC first so cuDNN's JIT-engine
    # plugins (cudnn_engines_runtime_compiled, cudnn_heuristic, …) can resolve
    # their transitive imports against an already-mapped module.
    "nvidia.nvjitlink",
    "nvidia.cuda_nvrtc",
    "nvidia.cuda_runtime",
    "nvidia.cublas",
    "nvidia.cudnn",
    "nvidia.cufft",
    "nvidia.curand",
    "nvidia.cusolver",
    "nvidia.cusparse",
)
_CUDA12_DLL_DIRS_REGISTERED = False
_CUDA12_PRELOADED_COUNT = 0
_CUDA12_WARNING: Optional[str] = None


def _cuda12_bin_dirs() -> list[Path]:
    import importlib.util

    bin_dirs: list[Path] = []
    for pkg in _CUDA12_PACKAGES:
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            continue
        if not spec or not spec.submodule_search_locations:
            continue
        for root in spec.submodule_search_locations:
            bin_dir = Path(root) / "bin"
            if bin_dir.is_dir():
                bin_dirs.append(bin_dir)
    return bin_dirs


def cuda12_runtime_available() -> bool:
    return bool(_cuda12_bin_dirs())


def cuda12_runtime_warning() -> Optional[str]:
    if cuda12_runtime_available():
        return None
    return (
        "CUDA selected, but CUDA 12 runtime wheels were not found. "
        "The wd-tagger will fall back to CPU."
    )


def _register_cuda12_dll_dirs(*, required: bool = False, ort=None) -> None:
    """Make CUDA 12 runtime DLLs visible to onnxruntime-gpu on Windows."""
    global _CUDA12_DLL_DIRS_REGISTERED, _CUDA12_PRELOADED_COUNT, _CUDA12_WARNING
    if _CUDA12_DLL_DIRS_REGISTERED or sys.platform != "win32":
        return
    _CUDA12_DLL_DIRS_REGISTERED = True

    import ctypes

    bin_dirs = _cuda12_bin_dirs()
    for bin_dir in bin_dirs:
        try:
            os.add_dll_directory(str(bin_dir))
        except OSError as exc:
            log.debug("add_dll_directory(%s) failed: %s", bin_dir, exc)

    if bin_dirs:
        os.environ["PATH"] = (
            os.pathsep.join(str(p) for p in bin_dirs)
            + os.pathsep
            + os.environ.get("PATH", "")
        )
        # Official ORT path since 1.21: preload NVIDIA DLLs from site-packages.
        if ort is not None and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
            except Exception as exc:
                log.debug("onnxruntime.preload_dlls failed: %s", exc)

    preloaded = 0
    for bin_dir in bin_dirs:
        for dll_path in bin_dir.glob("*.dll"):
            try:
                ctypes.CDLL(str(dll_path))
                preloaded += 1
            except OSError as exc:
                log.debug("preload %s failed: %s", dll_path, exc)

    _CUDA12_PRELOADED_COUNT = preloaded
    if preloaded:
        log.info("preloaded %d CUDA 12 DLLs for onnxruntime-gpu", preloaded)
    elif required:
        _CUDA12_WARNING = (
            "CUDA was requested but no CUDA 12 runtime wheels were found in this venv. "
            "The wd-tagger will fall back to CPU and indexing will be much slower."
        )
        log.warning(_CUDA12_WARNING)


class WDTaggerModel:
    """Lazy-loading wd-tagger with batched ONNX Runtime inference."""

    def __init__(
        self,
        repo_id: str = "SmilingWolf/wd-vit-tagger-v3",
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        hf_token: Optional[str] = None,
        device_backend: str = "auto",
    ):
        self.repo_id = repo_id
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.hf_token = hf_token
        self.device_backend = device_backend
        self._session = None
        self._labels: Optional[List[Tuple[str, int]]] = None
        self._input_size: int = 448
        self._input_name: Optional[str] = None
        self._active_providers: list[str] = []
        self._runtime_warnings: list[str] = []
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import csv

            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            _register_cuda12_dll_dirs(required=(self.device_backend == "cuda"), ort=ort)

            model_path = hf_hub_download(self.repo_id, "model.onnx", token=self.hf_token)
            labels_path = hf_hub_download(self.repo_id, "selected_tags.csv", token=self.hf_token)

            available = set(ort.get_available_providers())
            preferred: list[str] = []
            if self.device_backend == "cuda":
                preferred.append("CUDAExecutionProvider")
            elif self.device_backend == "dml":
                preferred.append("DmlExecutionProvider")
            elif self.device_backend == "auto":
                preferred.extend(["CUDAExecutionProvider", "DmlExecutionProvider"])

            providers = [p for p in preferred if p in available]
            providers.append("CPUExecutionProvider")

            self._session = ort.InferenceSession(model_path, providers=providers)
            self._active_providers = list(self._session.get_providers())
            input_meta = self._session.get_inputs()[0]
            self._input_name = input_meta.name
            self._input_size = input_meta.shape[1] or 448
            self._runtime_warnings = []

            if self.device_backend == "cuda" and "CUDAExecutionProvider" not in self._active_providers:
                self._runtime_warnings.append(
                    "CUDA selected, but wd-tagger is running on CPU. Check CUDA 12 runtimes and NVIDIA driver."
                )
            if _CUDA12_WARNING:
                self._runtime_warnings.append(_CUDA12_WARNING)

            labels: List[Tuple[str, int]] = []
            with open(labels_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labels.append((row["name"], int(row["category"])))
            self._labels = labels

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        side = self._input_size
        img = img.convert("RGB")
        ratio = side / max(img.size)
        new_size = (round(img.size[0] * ratio), round(img.size[1] * ratio))
        img = img.resize(new_size, Image.BICUBIC)
        background = Image.new("RGB", (side, side), (255, 255, 255))
        background.paste(img, ((side - new_size[0]) // 2, (side - new_size[1]) // 2))
        return np.array(background, dtype=np.float32)[:, :, ::-1]

    def _run_batch(self, images: List[Image.Image]) -> np.ndarray:
        self._ensure_loaded()
        assert self._session is not None and self._input_name is not None
        batch = np.stack([self._preprocess(img) for img in images], axis=0)
        try:
            return self._session.run(None, {self._input_name: batch})[0]
        except Exception:
            if len(images) <= 1:
                raise
            mid = len(images) // 2
            log.warning("wd-tagger batch of %d failed; retrying as smaller batches", len(images))
            return np.concatenate([self._run_batch(images[:mid]), self._run_batch(images[mid:])], axis=0)

    def active_providers(self) -> list[str]:
        return list(self._active_providers)

    def runtime_warnings(self) -> list[str]:
        return list(dict.fromkeys(self._runtime_warnings))

    def offload(self) -> None:
        """Free the ONNX session and its VRAM allocations.

        Required for the two-phase pipeline on 8 GB GPUs: after the tag pass
        we drop the tagger so SigLIP has the whole device to itself. The next
        `_ensure_loaded` call (e.g. on the next video's tag pass) re-creates
        the session lazily — model weights are cached on disk so reload is
        only ~1-2 s.
        """
        import gc
        with self._lock:
            self._session = None
            self._labels = None
            self._input_name = None
            self._active_providers = []
        gc.collect()

    def tag_image(self, image: Image.Image) -> List[Tuple[str, str, float]]:
        return self.tag_images([image])[0]

    def tag_images(self, images: List[Image.Image]) -> List[List[Tuple[str, str, float]]]:
        if not images:
            return []
        self._ensure_loaded()
        assert self._labels is not None

        outputs = self._run_batch(images)
        result: List[List[Tuple[str, str, float]]] = []
        for scores in outputs:
            out: List[Tuple[str, str, float]] = []
            for (name, category), score in zip(self._labels, scores):
                if category == 0 and score >= self.general_threshold:
                    out.append((name, _CATEGORY_NAMES[0], float(score)))
                elif category == 4 and score >= self.character_threshold:
                    out.append((name, _CATEGORY_NAMES[4], float(score)))
            out.sort(key=lambda t: t[2], reverse=True)
            result.append(out)
        return result

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        out = self._run_batch(images).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out /= np.clip(norms, 1e-12, None)
        return out
