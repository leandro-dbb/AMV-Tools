"""Finishing-model evaluation (chantier 4): upscale, interpolation, depth.

Reference data comes from the real episodes, at full 1080p:

    # 1080p stills / short frame runs at timestamps picked from a bench DB
    python tests/eval/eval_finish.py extract --db DATA/amv_tools.db --work W
    # x2 upscale of 540p (area-downscaled) stills, scored against the 1080p
    python tests/eval/eval_finish.py upscale --work W --model realesr-animevideov3
    # drop every other frame, regenerate it, score against the real one
    python tests/eval/eval_finish.py interp --work W --model rife-4.25
    # temporal flicker on static shots + timing
    python tests/eval/eval_finish.py depth --work W --model depth-anything-v2-small

Model weights: the app's finishing registry (``AMV_MODELS_DIR``) for the
current models; candidates from ``--weights`` (spandrel-loadable upscalers,
vs-rife flownets, Video-Depth-Anything checkpoints).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.eval.eval_models import _MemSampler, _backend, _cap_memory, _is_oom, _mem_report, _sync  # noqa: E402


# ── metrics ─────────────────────────────────────────────────────────────────
def _y(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    return 16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def psnr_y(a: np.ndarray, b: np.ndarray, border: int = 4) -> float:
    ya, yb = _y(a), _y(b)
    if border:
        ya, yb = ya[border:-border, border:-border], yb[border:-border, border:-border]
    mse = float(np.mean((ya - yb) ** 2))
    return 99.0 if mse < 1e-10 else 10 * math.log10(255.0 ** 2 / mse)


def ssim_y(a: np.ndarray, b: np.ndarray) -> float:
    import cv2
    x, y = _y(a), _y(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mx, my = cv2.GaussianBlur(x, k, 1.5), cv2.GaussianBlur(y, k, 1.5)
    sxx = cv2.GaussianBlur(x * x, k, 1.5) - mx * mx
    syy = cv2.GaussianBlur(y * y, k, 1.5) - my * my
    sxy = cv2.GaussianBlur(x * y, k, 1.5) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(m[5:-5, 5:-5].mean())


class _LPIPS:
    def __init__(self, backend: str):
        self.ok = False
        try:
            import lpips
            import torch
            self.dev = torch.device(backend)
            self.net = lpips.LPIPS(net="alex", verbose=False).eval().to(self.dev)
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            print("LPIPS unavailable:", exc, file=sys.stderr)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float | None:
        if not self.ok:
            return None
        import torch
        with torch.inference_mode():
            t = [torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().to(self.dev) / 127.5 - 1
                 for x in (a, b)]
            return float(self.net(*t).item())


# ── reference extraction ────────────────────────────────────────────────────
def _ffmpeg_frames(path: str, start_ms: int, n: int) -> list[np.ndarray]:
    """``n`` consecutive 1080p RGB frames from ``start_ms`` (CPU decode,
    exact frames, works for H.264 and 10-bit HEVC)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_ms / 1000:.3f}", "-i", path, "-frames:v", str(n),
           "-vf", "scale=1920:1080:flags=lanczos", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    fs = 1920 * 1080 * 3
    return [np.frombuffer(raw[i:i + fs], np.uint8).reshape(1080, 1920, 3) for i in range(0, len(raw) - fs + 1, fs)]


def cmd_extract(args) -> None:
    """Stills for upscale (luma spread incl. dark shots and flashes), runs of
    frames for interpolation (calm vs action by Search+ intensity) and depth
    (static shots by camera motion)."""
    work = Path(args.work)
    for d in ("stills", "runs"):
        (work / d).mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        vids = dict(conn.execute("SELECT id, filepath FROM videos").fetchall())
        cols = {r[1] for r in conn.execute("PRAGMA table_info(scene_metrics)")}
        print("scene_metrics columns:", sorted(cols))
        rows = conn.execute("SELECT s.id, s.video_id, s.start_ms, s.end_ms, m.intensity, m.motion_type "
                            "FROM scenes s JOIN scene_metrics m ON m.scene_id = s.id "
                            "WHERE s.parent_scene_id IS NULL").fetchall()
    rng = np.random.default_rng(4)
    man = {"stills": [], "runs": []}
    # stills: 12 per episode, scenes >= 1.5 s, spread over the episode
    for vid, path in vids.items():
        mine = [r for r in rows if r[1] == vid and r[3] - r[2] >= 1500]
        pick = sorted(rng.choice(len(mine), size=min(args.stills, len(mine)), replace=False))
        for i in pick:
            r = mine[i]
            t = (r[2] + r[3]) // 2
            f = _ffmpeg_frames(path, t, 1)[0]
            key = f"v{vid}_{t:07d}"
            np.save(work / "stills" / f"{key}.npy", f)
            man["stills"].append({"key": key, "video": vid, "ms": t, "luma": float(_y(f).mean())})
    # runs: per episode, the 3 calmest and 3 most intense shots >= 2 s, and
    # 3 static-camera shots for depth; 12 frames each
    for vid, path in vids.items():
        mine = [r for r in rows if r[1] == vid and r[3] - r[2] >= 2000 and r[4] is not None]
        mine.sort(key=lambda r: r[4])
        calm = mine[len(mine) // 10: len(mine) // 10 + 3]
        action = mine[-3:]
        static = [r for r in mine if (r[5] or "static") in ("static", "none", "fixed")][:3] or mine[:3]
        for label, group in (("calm", calm), ("action", action), ("static", static)):
            for r in group:
                frames = _ffmpeg_frames(path, r[2] + 200, args.run_len)
                key = f"{label}_v{vid}_{r[2]:07d}"
                np.save(work / "runs" / f"{key}.npy", np.stack(frames))
                man["runs"].append({"key": key, "label": label, "video": vid, "start_ms": r[2],
                                    "intensity": r[4], "camera": r[5], "n": len(frames)})
        print(f"video {vid}: done", flush=True)
    (work / "finish_manifest.json").write_text(json.dumps(man, indent=1))


# ── upscale ─────────────────────────────────────────────────────────────────
def _load_upscaler(model: str, backend: str, weights: str | None):
    """(callable NCHW float -> NCHW float, native scale)."""
    import torch
    dev = torch.device(backend)
    dtype = torch.float16 if backend in ("cuda", "mps") else torch.float32
    from backend.finish.upscale import UPSCALE_MODELS
    if model in UPSCALE_MODELS and not weights:
        from backend.finish.upscale import Upscaler
        from backend.models.device import DeviceInfo
        up = Upscaler(model, DeviceInfo(backend, backend, dev, dtype))
        return up.net, up.native_scale, dtype, dev
    if weights and weights.endswith(".onnx"):
        return _onnx_upscaler(weights, backend), 2, torch.float32, dev
    import spandrel
    desc = spandrel.ModelLoader().load_from_file(weights)
    net = desc.model.eval().to(dev)
    if desc.supports_half and dtype == torch.float16:
        net = net.half()
    else:
        dtype = torch.float32
    if backend == "cuda":
        net = net.to(memory_format=torch.channels_last)
    return net, desc.scale, dtype, dev


def _onnx_upscaler(path: str, backend: str):
    """ONNX-only x2 models (AnimeJaNai V3.1 SPAN, fp16 graph) through ONNX
    Runtime's CUDA provider, I/O bound to torch CUDA tensors (no host copy)."""
    import onnxruntime as ort
    import torch
    providers = (["CUDAExecutionProvider"] if backend == "cuda" else []) + ["CPUExecutionProvider"]
    sess = ort.InferenceSession(path, providers=providers)
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    half = "float16" in inp.type
    print(f"onnx {Path(path).name}: {inp.name} {inp.type} {inp.shape} -> {out.shape}, {sess.get_providers()}",
          file=sys.stderr, flush=True)

    def run(x):
        x = x.to(torch.float16 if half else torch.float32).contiguous()
        if backend != "cuda":
            y = sess.run([out.name], {inp.name: x.cpu().numpy()})[0]
            return torch.from_numpy(y).float()
        n, c, h, w = x.shape
        y = torch.empty((n, c, h * 2, w * 2), dtype=x.dtype, device=x.device)
        b = sess.io_binding()
        et = np.float16 if half else np.float32
        b.bind_input(inp.name, "cuda", 0, et, tuple(x.shape), x.data_ptr())
        b.bind_output(out.name, "cuda", 0, et, tuple(y.shape), y.data_ptr())
        torch.cuda.synchronize()
        sess.run_with_iobinding(b)
        return y
    return run


def cmd_upscale(args) -> None:
    import cv2
    import torch
    import torch.nn.functional as F
    backend = args.backend or _backend()
    _cap_memory(args.vram_gb, backend)
    work = Path(args.work)
    man = json.loads((work / "finish_manifest.json").read_text())
    res = {"model": args.model, "backend": backend, "vram_cap_gb": args.vram_gb}
    sampler = _MemSampler(backend).start()
    lp = _LPIPS(backend) if not args.no_lpips else None
    out_dir = work / "upscale_out" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.perf_counter()
        if args.model == "bicubic":
            fn, scale, dtype, dev = None, 2, None, None
        else:
            fn, scale, dtype, dev = _load_upscaler(args.model, backend, args.weights)
        res["load_s"] = round(time.perf_counter() - t0, 2)
        res["native_scale"] = scale
        scores, times = [], []
        for i, s in enumerate(man["stills"]):
            hr = np.load(work / "stills" / f"{s['key']}.npy")
            lr = cv2.resize(hr, (960, 540), interpolation=cv2.INTER_AREA)
            t1 = time.perf_counter()
            if fn is None:
                sr = cv2.resize(lr, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            else:
                with torch.inference_mode():
                    x = torch.from_numpy(lr).to(dev).permute(2, 0, 1)[None].to(dtype) / 255.0
                    if backend == "cuda":
                        x = x.contiguous(memory_format=torch.channels_last)
                    y = fn(x)
                    if scale > 2:
                        y = F.avg_pool2d(y.float(), scale // 2)
                    y = (y.float().clamp(0, 1)[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
                sr = y
            _sync(backend) if fn is not None else None
            if i > 0:
                times.append(time.perf_counter() - t1)
            sc = {"key": s["key"], "luma": s["luma"], "psnr": psnr_y(sr, hr), "ssim": ssim_y(sr, hr)}
            if lp is not None:
                sc["lpips"] = lp(sr, hr)
            scores.append(sc)
            if i < args.save:
                cv2.imwrite(str(out_dir / f"{s['key']}.png"), cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
        res["fps_540_to_1080"] = round(1 / float(np.median(times)), 2) if times else None
        for k in ("psnr", "ssim", "lpips"):
            vals = [x[k] for x in scores if x.get(k) is not None]
            res[k] = round(float(np.mean(vals)), 4) if vals else None
        dark = [x["psnr"] for x in scores if x["luma"] < 50]
        res["psnr_dark"] = round(float(np.mean(dark)), 3) if dark else None
        res["per_still"] = scores
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001
        res["ok"], res["oom"], res["error"] = False, _is_oom(exc), str(exc)[:300]
    sampler.stop()
    res.update(_mem_report(backend, sampler))
    print(json.dumps({k: v for k, v in res.items() if k != "per_still"}), flush=True)
    with open(work / "upscale_runs.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


# ── interpolation ───────────────────────────────────────────────────────────
def _classify(f0, f1, f2) -> str:
    """'dup' when the frame to rebuild repeats a neighbour (anime on twos/
    threes), 'static' when nothing moves, else 'inbetween'."""
    d01 = float(np.abs(f0.astype(np.int16) - f1).mean())
    d12 = float(np.abs(f1.astype(np.int16) - f2).mean())
    if d01 < 0.6 and d12 < 0.6:
        return "static"
    if d01 < 0.6 or d12 < 0.6:
        return "dup"
    return "inbetween"


def _load_interp(model: str, backend: str, weights: str | None):
    """Returns f(a, b, t) -> RGB uint8."""
    from backend.models.device import DeviceInfo
    import torch
    dev = torch.device(backend)
    info = DeviceInfo(backend, backend, dev, torch.float16 if backend == "cuda" else torch.float32)
    if model == "blend":
        return lambda a, b, t: ((1 - t) * a.astype(np.float32) + t * b).round().astype(np.uint8)
    if model == "film":
        # FILM (google-research, Apache-2.0), TorchScript port by dajes
        # (Apache-2.0): model(img0, img1, dt) on (1, 3, H, W) in [0, 1].
        net = torch.jit.load(weights, map_location="cpu").eval().to(dev)
        half = backend == "cuda"
        net = net.half() if half else net.float()

        def film(a, b, t):
            with torch.inference_mode():
                x = [torch.from_numpy(np.ascontiguousarray(f)).to(dev).permute(2, 0, 1)[None]
                     .to(torch.float16 if half else torch.float32) / 255.0 for f in (a, b)]
                dt = x[0].new_full((1, 1), t)
                y = net(x[0], x[1], dt)
                return (y.float().clamp(0, 1)[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
        return film
    if model.startswith("amt-"):
        return _amt(model, dev, weights)
    if model.startswith("gimm"):
        return _gimm(model, dev, weights)
    from backend.finish.interpolate import RifeInterpolator
    kw = {}
    if model != "rife-4.25":
        kw["model_id"] = model
    r = RifeInterpolator(info, **kw)

    def f(a, b, t):
        return r.at(r.make_pair(a, b), t)
    return f


def _rgb_in(f, dev):
    import torch
    return torch.from_numpy(np.ascontiguousarray(f)).to(dev).permute(2, 0, 1)[None].float() / 255.0


def _rgb_out(y):
    return (y.float().clamp(0, 1)[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()


def _pad16(xs, div: int = 16):
    import torch.nn.functional as F
    h, w = xs[0].shape[-2:]
    ph, pw = (-h) % div, (-w) % div
    pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
    return [F.pad(x, pad, mode="replicate") for x in xs], (pad[2], pad[2] + h, pad[0], pad[0] + w)


def _amt(model: str, dev, weights: str):
    """AMT (MCG-NKU, CC-BY-NC-4.0 code + weights) from its source tree
    (``AMT_SRC``): amt-s / amt-l / amt-g, fp32, scale 1, padding to 16 like
    the upstream demo (a 24 GB card keeps scale 1 at 1080p)."""
    import importlib
    import torch
    sys.path.insert(0, os.environ["AMT_SRC"])
    variant = model.split("-")[1].upper()
    mod = importlib.import_module(f"networks.AMT-{variant}")
    net = mod.Model(corr_radius=3, corr_lvls=4, num_flows=3 if variant == "S" else 5)
    import typing
    with torch.serialization.safe_globals([typing.OrderedDict]):
        ck = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(ck.get("state_dict", ck))
    net = net.eval().to(dev)

    def f(a, b, t):
        with torch.inference_mode():
            (x0, x1), (t0, t1, l0, l1) = _pad16([_rgb_in(a, dev), _rgb_in(b, dev)])
            embt = x0.new_full((1, 1, 1, 1), t)
            y = net(x0, x1, embt, scale_factor=1.0, eval=True)["imgt_pred"]
            return _rgb_out(y[..., t0:t1, l0:l1])
    return f


def _gimm(model: str, dev, weights: str):
    """GIMM-VFI-R-P (S-Lab License 1.0, non-commercial) from its source tree
    (``GIMM_SRC``; needs CuPy for softsplat). RAFT is loaded from
    ``pretrained_ckpt/raft-things.pth`` relative to ``GIMM_RUN``.
    ``gimm-r-p`` = full resolution, ``gimm-r-p-ds05`` = flow at half
    resolution (``ds_factor`` 0.5, upstream's setting for 2K)."""
    import torch
    from omegaconf import OmegaConf
    src = Path(os.environ["GIMM_SRC"])
    sys.path.insert(0, str(src / "src"))
    from models import create_model
    from utils.config import augment_arch_defaults
    cfg = OmegaConf.load(src / "configs/gimmvfi/gimmvfi_r_arb.yaml")
    cwd = os.getcwd()
    os.chdir(os.environ["GIMM_RUN"])
    try:
        net, _ = create_model(augment_arch_defaults(cfg.arch))
    finally:
        os.chdir(cwd)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["state_dict"], strict=True)
    net = net.eval().to(dev)
    ds = 0.5 if model.endswith("ds05") else 1.0

    def f(a, b, t):
        with torch.no_grad():
            (x0, x1), (t0, t1, l0, l1) = _pad16([_rgb_in(a, dev), _rgb_in(b, dev)], 32)
            xs = torch.cat((x0.unsqueeze(2), x1.unsqueeze(2)), dim=2)
            coord = [(net.sample_coord_input(1, xs.shape[-2:], [t], device=dev, upsample_ratio=ds), None)]
            ts = [t * torch.ones(1, device=dev)]
            y = net(xs, coord, t=ts, ds_factor=ds)["imgt_pred"][0]
            return _rgb_out(y[..., t0:t1, l0:l1])
    return f


def cmd_interp(args) -> None:
    backend = args.backend or _backend()
    _cap_memory(args.vram_gb, backend)
    work = Path(args.work)
    man = json.loads((work / "finish_manifest.json").read_text())
    res = {"model": args.model, "backend": backend, "vram_cap_gb": args.vram_gb}
    sampler = _MemSampler(backend).start()
    lp = _LPIPS(backend) if not args.no_lpips else None
    out_dir = work / "interp_out" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.perf_counter()
        fn = _load_interp(args.model, backend, args.weights)
        res["load_s"] = round(time.perf_counter() - t0, 2)
        per, times = [], []
        saved = 0
        for run in man["runs"]:
            if run["label"] not in ("calm", "action"):
                continue
            fr = np.load(work / "runs" / f"{run['key']}.npy")
            for i in range(0, len(fr) - 2, 2):
                f0, f1, f2 = fr[i], fr[i + 1], fr[i + 2]
                kind = _classify(f0, f1, f2)
                t1 = time.perf_counter()
                p = fn(f0, f2, 0.5)
                _sync(backend) if args.model != "blend" else None
                times.append(time.perf_counter() - t1)
                sc = {"run": run["key"], "i": i + 1, "label": run["label"], "kind": kind,
                      "psnr": psnr_y(p, f1), "ssim": ssim_y(p, f1)}
                if lp is not None and kind == "inbetween":
                    sc["lpips"] = lp(p, f1)
                per.append(sc)
                if kind == "inbetween" and saved < args.save and run["label"] == "action":
                    import cv2
                    cv2.imwrite(str(out_dir / f"{run['key']}_{i + 1:02d}.png"), cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
                    saved += 1
        res["fps_1080"] = round(1 / float(np.median(times[1:])), 2) if len(times) > 1 else None
        for label in ("calm", "action"):
            for kind in ("inbetween", "dup"):
                sel = [x for x in per if x["label"] == label and x["kind"] == kind]
                if not sel:
                    continue
                key = f"{label}_{kind}"
                res[key] = {"n": len(sel), "psnr": round(float(np.mean([x["psnr"] for x in sel])), 3),
                            "ssim": round(float(np.mean([x["ssim"] for x in sel])), 4)}
                lps = [x["lpips"] for x in sel if x.get("lpips") is not None]
                if lps:
                    res[key]["lpips"] = round(float(np.mean(lps)), 4)
        res["per"] = per
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001
        res["ok"], res["oom"], res["error"] = False, _is_oom(exc), str(exc)[:300]
    sampler.stop()
    res.update(_mem_report(backend, sampler))
    print(json.dumps({k: v for k, v in res.items() if k != "per"}), flush=True)
    with open(work / "interp_runs.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


# ── depth ───────────────────────────────────────────────────────────────────
def _depth_fn(model: str, backend: str, input_size: int, weights: str | None):
    """Returns f(list of frames) -> list of disparity maps (model res)."""
    import torch
    from backend.models.device import DeviceInfo
    dev = torch.device(backend)
    info = DeviceInfo(backend, backend, dev, torch.float16 if backend == "cuda" else torch.float32)
    if model.startswith("vda"):
        # Video-Depth-Anything (Apache-2.0 code + Small weights), from its
        # source tree (VDA_SRC): 32-frame windows with temporal attention.
        sys.path.insert(0, os.environ["VDA_SRC"])
        from video_depth_anything.video_depth import VideoDepthAnything
        net = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        net = net.to(dev).eval()

        def run(frames):
            d, _ = net.infer_video_depth(np.stack(frames), 24, input_size=input_size, device=backend,
                                         fp32=backend != "cuda")
            return [cv2_resize_like(x) for x in d]
        return run
    if model.startswith("video-depth-anything"):
        # Small from the registry; Base / Large (CC-BY-NC-4.0, chantier 4b)
        # from --weights
        from backend.finish.vda import VideoDepthEstimator
        vest = VideoDepthEstimator(info, input_size=input_size, model_id=model, weights=weights)
        return lambda frames: [d for _, d in vest.predict_stream(frames)]
    from backend.finish.depth import DepthEstimator
    kw = {} if model == "depth-anything-v2-small" else {"model_id": model}
    est = DepthEstimator(info, input_size=input_size, **kw)
    if backend == "cuda":
        def run(frames):
            out = []
            for i in range(0, len(frames), 8):
                out += est.predict_batch(frames[i:i + 8])
            return out
        return run
    return lambda frames: [d for _, d in est.predict_stream(frames)]


def cv2_resize_like(d: np.ndarray, max_side: int = 518) -> np.ndarray:
    import cv2
    h, w = d.shape
    s = max_side / max(h, w)
    return cv2.resize(d.astype(np.float32), (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _flicker(stack: np.ndarray, frames: list[np.ndarray]) -> list[float]:
    """Mean |d_t - d_t-1| (in % of the clip's depth range) on the pixels
    whose colour did not change between the two frames."""
    import cv2
    st = np.stack([cv2.resize(np.asarray(d, np.float32), (480, 270)) for d in stack])
    lo, hi = np.percentile(st, [0.5, 99.5])
    st = np.clip((st - lo) / max(1e-6, hi - lo), 0, 1)
    sm = np.stack([cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA) for f in frames]).astype(np.int16)
    out = []
    for i in range(1, len(st)):
        still = np.abs(sm[i] - sm[i - 1]).max(axis=-1) < 4
        if still.mean() < 0.3:
            continue
        out.append(float(np.abs(st[i] - st[i - 1])[still].mean()) * 100)
    return out


def cmd_depth(args) -> None:
    import cv2
    from backend.finish.depth import stabilize_depth
    backend = args.backend or _backend()
    _cap_memory(args.vram_gb, backend)
    work = Path(args.work)
    man = json.loads((work / "finish_manifest.json").read_text())
    res = {"model": args.model, "backend": backend, "input_size": args.input_size, "vram_cap_gb": args.vram_gb}
    sampler = _MemSampler(backend).start()
    out_dir = work / "depth_out" / f"{args.model}_{args.input_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.perf_counter()
        fn = _depth_fn(args.model, backend, args.input_size, args.weights)
        res["load_s"] = round(time.perf_counter() - t0, 2)
        # speed: the action runs back to back (~5 s of 1080p, like the
        # bench's finishing clip); one warm-up pass first
        action = [f for r in man["runs"] if r["label"] == "action" for f in np.load(work / "runs" / f"{r['key']}.npy")]
        fn(action[:16])
        _sync(backend)
        t1 = time.perf_counter()
        fn(action)
        _sync(backend)
        res["fps_1080"] = round(len(action) / (time.perf_counter() - t1), 2)
        fl_raw, fl_app, fl_align, t_stab = [], [], [], 0.0
        saved = 0
        for run in man["runs"]:
            if run["label"] not in ("static", "calm"):
                continue
            frames = list(np.load(work / "runs" / f"{run['key']}.npy"))
            ds = [np.asarray(d, np.float32) for d in fn(frames)]
            fl_raw += _flicker(np.stack([cv2.resize(d, ds[0].shape[::-1]) for d in ds]), frames)
            smalls = [cv2.resize(f, (ds[0].shape[1], ds[0].shape[0]), interpolation=cv2.INTER_AREA) for f in frames]
            ds = [cv2.resize(d, (ds[0].shape[1], ds[0].shape[0])) for d in ds]
            t2 = time.perf_counter()
            app = stabilize_depth(ds, smalls, cuts=set(), smoothing=0.5)
            t_stab += time.perf_counter() - t2
            fl_app += _flicker(app, frames)
            fl_align += _flicker(stabilize_depth(ds, smalls, cuts=set(), smoothing=0.0), frames)
            if saved < args.save:
                strip = np.concatenate([(cv2.resize(app[k], (480, 270)) * 255).astype(np.uint8)
                                        for k in (0, len(app) // 2, len(app) - 1)], axis=1)
                rgb = np.concatenate([cv2.resize(frames[k], (480, 270)) for k in (0, len(app) // 2, len(app) - 1)],
                                     axis=1)
                cv2.imwrite(str(out_dir / f"{run['key']}.png"),
                            np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                            cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)], axis=0))
                saved += 1
        for name, vals in (("flicker_raw", fl_raw), ("flicker_align", fl_align), ("flicker_app", fl_app)):
            res[name] = round(float(np.mean(vals)), 3) if vals else None
        res["flicker_pairs"] = len(fl_raw)
        res["stabilize_s_per_frame"] = round(t_stab / max(1, len(fl_raw)), 4)
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        res["ok"], res["oom"], res["error"] = False, _is_oom(exc), str(exc)[:300]
    sampler.stop()
    res.update(_mem_report(backend, sampler))
    print(json.dumps(res), flush=True)
    with open(work / "depth_runs.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--db", required=True)
    e.add_argument("--work", required=True)
    e.add_argument("--stills", type=int, default=12)
    e.add_argument("--run-len", type=int, default=13)
    for name in ("upscale", "interp", "depth"):
        s = sub.add_parser(name)
        s.add_argument("--work", required=True)
        s.add_argument("--model", required=True)
        s.add_argument("--weights")
        s.add_argument("--backend")
        s.add_argument("--vram-gb", type=float)
        s.add_argument("--save", type=int, default=6)
        s.add_argument("--no-lpips", action="store_true")
        s.add_argument("--input-size", type=int, default=518)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    {"extract": cmd_extract, "upscale": cmd_upscale, "interp": cmd_interp, "depth": cmd_depth}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
