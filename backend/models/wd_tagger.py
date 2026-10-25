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


# Core ML specialises a program per input shape. The wd-tagger exports a
# dynamic batch dimension; left dynamic, ORT splits the ViT into ~38 Core ML
# partitions and the GPU compile of that dynamic MLProgram aborts the whole
# process on Apple Silicon (MPSGraph "original module failed verification",
# measured with onnxruntime 1.26 / macOS 26 — the sidecar died on the first
# tag batch). Pinning the batch makes the whole graph a single partition
# (467/467 nodes on Core ML); the last batch of a video is padded.
COREML_TAGGER_BATCH = 8


def _coreml_provider_entry(cache_key: str = "") -> tuple:
    """CoreML EP with its measured-best options on Apple Silicon.

    MLProgram = the modern Core ML format (better op coverage; the legacy
    NeuralNetwork format ran the tagger at 0.2 img/s). MLComputeUnits:
    CPUAndGPU beat ALL and CPUAndNeuralEngine on both ONNX models (M1 Pro:
    wd-tagger 17.6 vs 13.1 vs 6.5 img/s, ISNet 2.8 vs 2.6 vs 1.6 img/s) and
    compiles faster. ModelCacheDirectory persists the compiled model —
    without it every session re-creation pays the full Core ML compile
    (~2 min for the tagger) again.
    """
    options = {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}
    try:
        from ..paths import user_data_dir
        # ORT keys the cache on the model, not on input-shape overrides: a
        # program compiled for another batch size would be reused and fail
        # at run time, so fixed-shape sessions get their own subdirectory.
        cache = user_data_dir() / "coreml_cache"
        if cache_key:
            cache = cache / cache_key
        cache.mkdir(parents=True, exist_ok=True)
        options["ModelCacheDirectory"] = str(cache)
    except Exception:
        pass
    return ("CoreMLExecutionProvider", options)


def build_onnx_providers(device_backend: str, ort, coreml_cache_key: str = "") -> list:
    """Provider list for InferenceSession, best hardware EP first.

    CoreML is included on macOS for the 'mps'/'auto' backends — without it
    the ONNX models (wd-tagger, anime ISNet) run pure-CPU on Apple Silicon,
    which is ~20× slower than the ANE/GPU path for the tagging pass.
    """
    available = set(ort.get_available_providers())
    preferred: list = []
    if device_backend == "cuda":
        preferred.append("CUDAExecutionProvider")
    elif device_backend == "dml":
        preferred.append("DmlExecutionProvider")
    elif device_backend == "mps":
        preferred.append(_coreml_provider_entry(coreml_cache_key))
    elif device_backend == "auto":
        preferred.extend(["CUDAExecutionProvider", "DmlExecutionProvider"])
        if sys.platform == "darwin":
            preferred.append(_coreml_provider_entry(coreml_cache_key))
    providers = [p for p in preferred
                 if (p[0] if isinstance(p, tuple) else p) in available]
    providers.append("CPUExecutionProvider")
    return providers


def make_onnx_session(ort, model_path: str, providers: list, sess_options=None):
    """InferenceSession with graceful degradation.

    Older onnxruntime wheels reject the CoreML string options, and a broken
    EP install shouldn't take indexing down with it — retry with the bare
    provider names, then CPU-only as the last resort.
    """
    try:
        return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    except Exception as exc:
        stripped = [p[0] if isinstance(p, tuple) else p for p in providers]
        log.warning("InferenceSession with %s failed (%s); retrying without provider options",
                    stripped, exc)
        try:
            return ort.InferenceSession(model_path, sess_options=sess_options, providers=stripped)
        except Exception as exc2:
            log.warning("hardware EPs unavailable (%s); falling back to CPU", exc2)
            return ort.InferenceSession(model_path, sess_options=sess_options,
                                        providers=["CPUExecutionProvider"])


def _uses_coreml(providers: list) -> bool:
    return any((p[0] if isinstance(p, tuple) else p) == "CoreMLExecutionProvider" for p in providers)


# Files each runtime needs from the model repo.
ONNX_FILES = ("model.onnx", "selected_tags.csv")
TORCH_FILES = ("config.json", "model.safetensors", "selected_tags.csv")


def weight_files(device_backend: Optional[str]) -> tuple[str, ...]:
    """Repo files the tagger downloads on this backend."""
    return TORCH_FILES if device_backend == "mps" else ONNX_FILES


def _read_labels(path: str) -> List[Tuple[str, int]]:
    import csv

    with open(path, "r", encoding="utf-8") as f:
        return [(row["name"], int(row["category"])) for row in csv.DictReader(f)]


class _TorchTagger:
    """The same wd-tagger v3 checkpoint through timm on PyTorch MPS.

    Exposes the ``run(None, {name: batch})`` contract of an ONNX session
    (BGR, 0-255, NHWC float32 in; sigmoid scores out) so batching and tag
    decoding stay shared. Why not Core ML on Apple Silicon: building the
    Core ML session holds the GIL for ~32 s (2 min the first time) — the
    whole sidecar freezes — caches several GB of compiled programs, and only
    works with a pinned batch. The timm checkpoint loads in ~1.5 s, runs as
    fast (M1 Pro: 17 img/s for both) and matches the ONNX outputs (fp16: max
    |Δscore| 0.003, identical top-10 and tag sets on real episode frames).
    """

    def __init__(self, config_path: str, weights_path: str, device="mps"):
        import json

        import timm
        import torch
        from safetensors.torch import load_file

        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        model = timm.create_model(cfg["architecture"], num_classes=cfg["num_classes"],
                                  pretrained=False, **cfg.get("model_args", {}))
        model.load_state_dict(load_file(weights_path))
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.model = model.eval().to(self.device, self.dtype)
        mean = (cfg.get("pretrained_cfg") or {}).get("mean") or [0.5, 0.5, 0.5]
        std = (cfg.get("pretrained_cfg") or {}).get("std") or [0.5, 0.5, 0.5]
        # Input arrives BGR (the ONNX convention); the checkpoint was trained
        # on BGR too (SmilingWolf's wdv3_timm reference flips RGB -> BGR), so
        # only the scaling/normalisation is applied here.
        self._mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1) * 255.0
        self._std = torch.tensor(std, device=self.device).view(1, 3, 1, 1) * 255.0
        # Core ML GPU worker, 1.4x faster on an M1 Pro (see ``coreml_tagger``);
        # MPS until up. The automatic tuning may have measured MPS faster here.
        from .. import autotune
        from . import coreml_tagger
        self.paths = (config_path, weights_path)
        self._cml = (coreml_tagger.start(config_path, weights_path)
                     if autotune.value("tagger_engine", "coreml") == "coreml" else None)

    def _run_coreml(self, x: np.ndarray) -> np.ndarray:
        from .coreml_tagger import BATCH
        ids = [self._cml.submit([x[k:k + BATCH]]) for k in range(0, len(x), BATCH)]
        return np.concatenate([self._cml.result(i)[0] for i in ids])

    def out_of_process(self) -> bool:
        return self._cml is not None and self._cml.usable()

    def close(self) -> None:
        if getattr(self, "_cml", None) is not None:
            self._cml.close()
            self._cml = None

    def run(self, _outputs, feed: dict):
        import torch

        x = np.ascontiguousarray(next(iter(feed.values())), dtype=np.float32)
        if self._cml is not None and self._cml.usable():
            try:
                return [self._run_coreml(x)]
            except Exception as exc:
                log.warning("Core ML wd-tagger stopped (%s); back to MPS", exc)
                self._cml = None
        elif self._cml is not None and self._cml.failed:
            log.warning("Core ML wd-tagger unavailable (%s); MPS", self._cml.failed)
            self._cml = None
        with torch.inference_mode():
            # Blocking copy: see SigLIP2Model._to_device.
            t = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2)
            t = ((t - self._mean) / self._std).to(self.dtype)
            y = torch.sigmoid(self.model(t).float())
            return [y.cpu().numpy()]


_PRE_POOL = None


def _preprocess_pool():
    global _PRE_POOL
    if _PRE_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _PRE_POOL = ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) // 2)),
                                       thread_name_prefix="wd-pre")
    return _PRE_POOL


class WDTaggerModel:
    """Lazy-loading wd-tagger with batched ONNX Runtime inference
    (PyTorch MPS on Apple Silicon)."""

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
        self._static_batch: Optional[int] = None
        self._active_providers: list[str] = []
        self._runtime_warnings: list[str] = []
        self._lock = threading.Lock()
        # CUDA: the same checkpoint through PyTorch fp16 (see _load_torch).
        self._torch_model = None
        self._torch_mean = self._torch_std = None
        self._torch_failed = False

    def _ensure_loaded(self):
        if self._session is not None or self._torch_model is not None:
            return
        with self._lock:
            if self._session is not None or self._torch_model is not None:
                return

            if self.device_backend == "cuda" and not self._torch_failed and self._load_torch():
                return

            import onnxruntime as ort

            from ..model_download import download_model_files

            _register_cuda12_dll_dirs(required=(self.device_backend == "cuda"), ort=ort)

            if self.device_backend == "mps" and self._load_torch_mps(download_model_files):
                return

            model_path, labels_path = download_model_files(
                self.repo_id,
                ONNX_FILES,
                token=self.hf_token,
            )

            providers = build_onnx_providers(self.device_backend, ort,
                                             coreml_cache_key=f"wd-tagger-b{COREML_TAGGER_BATCH}")
            sess_options = None
            if _uses_coreml(providers):
                sess_options = ort.SessionOptions()
                sess_options.add_free_dimension_override_by_name("batch_size", COREML_TAGGER_BATCH)
            self._session = make_onnx_session(ort, model_path, providers, sess_options)
            input_meta = self._session.get_inputs()[0]
            batch_dim = input_meta.shape[0]
            self._static_batch = batch_dim if isinstance(batch_dim, int) and batch_dim > 0 else None
            if ("CoreMLExecutionProvider" in self._session.get_providers()
                    and self._static_batch is None):
                # The override did not apply (different dim name): a dynamic
                # batch on Core ML can abort the process — stay on CPU.
                log.warning("wd-tagger batch dimension is not static; Core ML disabled for it")
                self._session = make_onnx_session(ort, model_path, ["CPUExecutionProvider"])
            self._active_providers = list(self._session.get_providers())
            self._input_name = input_meta.name
            self._input_size = input_meta.shape[1] or 448
            self._runtime_warnings = []

            if self.device_backend == "cuda" and "CUDAExecutionProvider" not in self._active_providers:
                self._runtime_warnings.append(
                    "CUDA selected, but wd-tagger is running on CPU. Check CUDA 12 runtimes and NVIDIA driver."
                )
            if self.device_backend == "mps" and "CoreMLExecutionProvider" not in self._active_providers:
                self._runtime_warnings.append(
                    "Apple Silicon selected, but wd-tagger is running on CPU (Core ML EP unavailable). "
                    "Reinstall the MPS backend from Settings → Models."
                )
            if _CUDA12_WARNING:
                self._runtime_warnings.append(_CUDA12_WARNING)

            self._labels = _read_labels(labels_path)

    def _load_torch_mps(self, download_model_files) -> bool:
        """Apple Silicon path (see :class:`_TorchTagger`). False → the
        caller falls back to ONNX Runtime + Core ML."""
        try:
            config_path, weights_path, labels_path = download_model_files(
                self.repo_id, TORCH_FILES, token=self.hf_token,
            )
            self._session = _TorchTagger(config_path, weights_path)
        except Exception as exc:
            log.warning("wd-tagger on MPS unavailable (%s); using ONNX Runtime", exc)
            self._session = None
            return False
        self._input_name = "input"
        self._input_size = 448
        self._static_batch = None
        self._active_providers = ["MPS"]
        self._runtime_warnings = []
        self._labels = _read_labels(labels_path)
        return True

    def _load_torch(self) -> bool:
        """Run the tagger with PyTorch fp16 on CUDA instead of ONNX Runtime.

        SmilingWolf publishes the same checkpoint as timm safetensors next to
        the ONNX file. ORT's CUDA EP only runs it in fp32 (~90 img/s on an RTX
        3090, with a dozen host round-trips in the graph); timm + fp16 + SDPA
        does ~280 img/s with the same tag sets (benchmark in
        docs/PERFORMANCE-NVIDIA.md). Any failure — no timm, offline without
        the safetensors file, old GPU — falls back to the ONNX session."""
        try:
            import json

            import timm
            import torch
            from safetensors.torch import load_file

            from ..model_download import download_model_files

            if not torch.cuda.is_available():
                return False
            cfg_path, weights_path, labels_path = download_model_files(
                self.repo_id, ("config.json", "model.safetensors", "selected_tags.csv"),
                token=self.hf_token,
            )
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Built on the meta device and filled straight from the GPU copy
            # of the weights: skips timm's random init (~1.5 s) and a host copy.
            with torch.device("meta"):
                model = timm.create_model(cfg["architecture"], pretrained=False,
                                          num_classes=cfg["num_classes"], **cfg.get("model_args", {}))
            model.load_state_dict(load_file(weights_path, device="cuda"), assign=True)
            if any(t.is_meta for t in model.buffers()):
                # Non-persistent buffers (SwinV2's relative-position tables)
                # are not in the checkpoint: build this one the slow way.
                model = timm.create_model(cfg["architecture"], pretrained=False,
                                          num_classes=cfg["num_classes"], **cfg.get("model_args", {}))
                model.load_state_dict(load_file(weights_path))
                model = model.to("cuda")
            pcfg = cfg.get("pretrained_cfg", {})
            dtype = torch.float16 if torch.cuda.get_device_capability()[0] >= 7 else torch.float32
            self._torch_model = model.eval().to(dtype)
            self._input_size = int((pcfg.get("input_size") or [3, 448, 448])[-1])
            self._torch_mean = torch.tensor(pcfg.get("mean", [0.5] * 3), device="cuda").view(1, 3, 1, 1)
            self._torch_std = torch.tensor(pcfg.get("std", [0.5] * 3), device="cuda").view(1, 3, 1, 1)
            self._labels = _read_labels(labels_path)
            self._active_providers = [f"PyTorch CUDA ({str(dtype).replace('torch.', '')})"]
            self._runtime_warnings = []
            log.info("wd-tagger running through PyTorch %s on CUDA", dtype)
            return True
        except Exception as exc:
            self._torch_failed = True
            self._torch_model = None
            log.warning("wd-tagger PyTorch/CUDA path unavailable (%s); using ONNX Runtime", exc)
            return False

    def _preprocess_u8(self, img: Image.Image) -> np.ndarray:
        """Letterbox on white to a square, BGR uint8 (H, W, 3)."""
        side = self._input_size
        img = img.convert("RGB")
        ratio = side / max(img.size)
        new_size = (round(img.size[0] * ratio), round(img.size[1] * ratio))
        img = img.resize(new_size, Image.BICUBIC)
        background = Image.new("RGB", (side, side), (255, 255, 255))
        background.paste(img, ((side - new_size[0]) // 2, (side - new_size[1]) // 2))
        return np.asarray(background)[:, :, ::-1]

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        return self._preprocess_u8(img).astype(np.float32)

    def _letterbox_gpu(self, images: List[Image.Image]):
        """``_preprocess_u8`` for a batch of same-size RGB frames, on the GPU
        (antialiased bicubic like PIL's, rounded to uint8, white letterbox,
        BGR): (N, 3, side, side) float 0..255, or None for mixed sizes."""
        import torch
        import torch.nn.functional as F

        if not images or any(im.size != images[0].size or im.mode != "RGB" for im in images):
            return None
        side = self._input_size
        w, h = images[0].size
        ratio = side / max(w, h)
        nw, nh = round(w * ratio), round(h * ratio)
        x = torch.from_numpy(np.stack([np.asarray(im) for im in images])).to("cuda", non_blocking=True)
        x = x.permute(0, 3, 1, 2).float()
        if (nw, nh) != (w, h):
            x = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
            x = x.round_().clamp_(0, 255)
        out = torch.full((len(images), 3, side, side), 255.0, device=x.device)
        top, left = (side - nh) // 2, (side - nw) // 2
        out[:, :, top:top + nh, left:left + nw] = x
        return out.flip(1)   # RGB -> BGR

    def _run_torch(self, images: List[Image.Image], chunk: int = 0) -> np.ndarray:
        import torch

        # 4 chunks per call (8 to 32 images): big enough batches for the GPU,
        # small enough to overlap the letterboxing of the next one
        chunk = chunk or min(32, max(8, len(images) // 4))
        model = self._torch_model
        dtype = next(model.parameters()).dtype
        outs = []
        with torch.inference_mode():
            # Chunks: kernels are queued asynchronously, so the letterboxing
            # of chunk k+1 (PIL, on a few threads — it releases the GIL)
            # overlaps the GPU work of chunk k. One sync at the end.
            for i in range(0, len(images), chunk):
                part = images[i:i + chunk]
                x = self._letterbox_gpu(part) if os.environ.get("AMV_TAGGER_GPU_PRE", "1") != "0" else None
                if x is None:
                    batch = np.stack(list(_preprocess_pool().map(self._preprocess_u8, part)))
                    x = torch.from_numpy(batch).to("cuda").permute(0, 3, 1, 2).float()
                x = ((x / 255.0 - self._torch_mean) / self._torch_std).to(dtype)
                outs.append(model(x))
            return torch.sigmoid(torch.cat(outs).float()).cpu().numpy()

    def _run_batch(self, images: List[Image.Image]) -> np.ndarray:
        self._ensure_loaded()
        if self._torch_model is None and self._static_batch:
            return self._run_static(images, self._static_batch)
        try:
            if self._torch_model is not None:
                return self._run_torch(images)
            assert self._session is not None and self._input_name is not None
            if isinstance(self._session, _TorchTagger):
                # PIL letterboxing releases the GIL: a few threads keep the
                # GPU fed (same arrays)
                batch = np.stack(list(_preprocess_pool().map(self._preprocess, images)), axis=0)
            else:
                batch = np.stack([self._preprocess(img) for img in images], axis=0)
            return self._session.run(None, {self._input_name: batch})[0]
        except Exception:
            if len(images) <= 1:
                raise
            mid = len(images) // 2
            log.warning("wd-tagger batch of %d failed; retrying as smaller batches", len(images))
            return np.concatenate([self._run_batch(images[:mid]), self._run_batch(images[mid:])], axis=0)

    def _run_static(self, images: List[Image.Image], size: int) -> np.ndarray:
        """Fixed-shape session: run in chunks of exactly ``size`` images,
        padding the last chunk with copies of its final image."""
        outs = []
        for i in range(0, len(images), size):
            chunk = np.stack([self._preprocess(img) for img in images[i:i + size]], axis=0)
            n = len(chunk)
            if n < size:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], size - n, axis=0)], axis=0)
            outs.append(self._session.run(None, {self._input_name: chunk})[0][:n])
        return np.concatenate(outs, axis=0)

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
            had_torch = self._torch_model is not None
            if isinstance(self._session, _TorchTagger):
                self._session.close()
            self._torch_model = None
            self._session = None
            self._labels = None
            self._input_name = None
            self._static_batch = None
            self._active_providers = []
        gc.collect()
        if had_torch or self.device_backend == "mps":
            from .device import empty_device_cache
            empty_device_cache("mps" if self.device_backend == "mps" else "cuda")

    def tag_image(self, image: Image.Image) -> List[Tuple[str, str, float]]:
        return self.tag_images([image])[0]

    def tag_images(self, images: List[Image.Image]) -> List[List[Tuple[str, str, float]]]:
        if not images:
            return []
        self._ensure_loaded()
        assert self._labels is not None

        outputs = self._run_batch(images)
        # Vectorised threshold over the ~10k labels (was a Python loop per
        # label per image); same tags, same order.
        cats = np.fromiter((c for _, c in self._labels), dtype=np.int16, count=len(self._labels))
        result: List[List[Tuple[str, str, float]]] = []
        for scores in outputs:
            scores = np.asarray(scores)[:len(cats)]
            keep = np.flatnonzero(((cats == 0) & (scores >= self.general_threshold))
                                  | ((cats == 4) & (scores >= self.character_threshold)))
            out = [(self._labels[i][0], _CATEGORY_NAMES[int(cats[i])], float(scores[i])) for i in keep]
            out.sort(key=lambda t: t[2], reverse=True)
            result.append(out)
        return result

    def runs_out_of_process(self) -> bool:
        """True when inference happens in another process (Core ML worker),
        so another model can use this one's thread time."""
        sess = self._session
        return isinstance(sess, _TorchTagger) and sess.out_of_process()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        out = self._run_batch(images).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out /= np.clip(norms, 1e-12, None)
        return out
