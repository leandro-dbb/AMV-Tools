"""Mask Mode evaluation on real anime shots (chantier roto, 0.5).

Not a unit test: a measurement tool run by hand on a GPU machine. Media and
ground truth never go in the repo; everything lives under ``--root``:

    <root>/shots.json                 shot list (id, src key, start s, dur s, category)
    <root>/shots/<id>/frames/*.png    frames at the Mask Mode working size
    <root>/gt/<id>/<idx>.png          reference masks on the key frames (0/255)
    <root>/gt_fix.json                manual prompts used to correct the references
    <root>/runs/<method>/<id>/*.png   8-bit alpha of every frame + stats.json

Sub-commands (``python -m tests.eval.roto_eval <cmd> --root R ...``):

    extract   decode each shot (``--src key=/path/video``) at ``--max-dim``
    propose   build candidate references on the key frames (SAM 2.1 large on
              boxes from the anime matte, or the prompts of gt_fix.json) and
              write review sheets (frame / mask / overlay) to <root>/review
    run       run one method on every shot (``--method``)
    score     metrics of every method present in runs/ -> scores.json + table
    boards    qualitative boards (checkerboard + colour composite) per shot

Metrics (per key frame, then averaged per shot and per category):
    J        IoU of alpha > 0.5 with the reference
    F        boundary F-measure, DAVIS tolerance (0.8 % of the diagonal)
    edge     mean |alpha - ref| in a 8 px band around the reference contour (x100)
    flicker  flow-compensated temporal change of the alpha, relative to the
             matte area (%) — computed on every frame, no reference needed
    fail     key frames with J < 0.5 (character lost / swapped)
    bg_band  background kept in the mask along the outline (hair gaps, halo):
             mean alpha of the reference-background pixels 2 to BAND_PX px
             from the reference contour (%), on the *exported* alpha (the
             first 2 px are not scored: the reference is only that precise)
    bg_in    background kept inside the character's envelope (between the arms,
             under the chin, big hair holes): mean alpha of the reference-
             background pixels enclosed by the closed silhouette, farther than
             BAND_PX from the contour (%)
    eaten    character eaten along the outline: mean (1 - alpha) of the
             reference-character pixels 2 to BAND_PX px inside the contour (%)
    contam   colour contamination of the export: share of background colour
             left in the semi-transparent edge pixels of the exported RGB
             (projection on the local background-foreground colour axis,
             alpha-weighted, %)
    ms/frame wall time per frame after warm-up, peak VRAM (nvidia-smi), peak RSS
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

N_KEYFRAMES = 6  # one is the click / seed frame (see seed_frame), the other 5 are scored


def key_frames(n: int) -> list[int]:
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, N_KEYFRAMES)})


def seed_frame(root: Path, sid: str, n: int) -> int:
    """Key frame the simulated user clicks on: the one where the reference
    covers the most pixels (a user clicks where the character is visible —
    in an occlusion shot that is not frame 0)."""
    best, area = 0, -1.0
    for k in key_frames(n):
        if not (root / "gt" / sid / f"{k:05d}.png").exists():
            continue
        a = float((read_alpha(root / "gt" / sid / f"{k:05d}.png") > 0.5).mean())
        if a > area + 0.02:  # earliest wins unless clearly smaller
            best, area = k, a
    return best


def scored_frames(root: Path, sid: str, n: int) -> list[int]:
    """Every frame with a reference except the seed frame: the 6 key frames,
    plus the extra references added where a shot needs a closer look
    (roto stable: blur frames of s13, the cigarette of b1...)."""
    seed = seed_frame(root, sid, n)
    have = sorted(int(p.stem) for p in (root / "gt" / sid).glob("*.png") if p.stem.isdigit())
    return [k for k in have if k != seed and k < n]


def load_shots(root: Path) -> list[dict]:
    return json.loads((root / "shots.json").read_text())


def frame_paths(root: Path, sid: str) -> list[Path]:
    return sorted((root / "shots" / sid / "frames").glob("*.png"))


def read_rgb(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"))


def read_alpha(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def save_alpha(p: Path, a: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)).save(p)


# ── extract ────────────────────────────────────────────────────────────────
def cmd_extract(args) -> None:
    from backend.export.ffmpeg import extract_png_frames
    from backend.indexing.cuts import probe_video

    root = Path(args.root)
    srcs = dict(s.split("=", 1) for s in args.src)
    for shot in load_shots(root):
        out = root / "shots" / shot["id"] / "frames"
        if out.exists() and any(out.glob("*.png")):
            continue
        out.mkdir(parents=True, exist_ok=True)
        video = srcs[shot["src"]]
        fps = float(probe_video(video).get("fps") or 24.0)
        start = int(shot["start"] * 1000)
        end = start + int(shot["dur"] * 1000)
        paths = extract_png_frames(video, start, end, out / "%05d.png", fps=fps, max_dim=args.max_dim)
        print(shot["id"], len(paths), "frames", flush=True)


# ── models (eval-side loaders) ─────────────────────────────────────────────
def _device():
    from backend.models.device import detect_device
    return detect_device()


class _VramPoll:
    """Peak GPU memory of this process (nvidia-smi), covers torch + onnxruntime."""

    def __init__(self):
        self.peak_mb = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in out.splitlines():
                    p, _, mb = line.partition(",")
                    if p.strip() == pid:
                        self.peak_mb = max(self.peak_mb, int(mb.strip()))
            except Exception:
                pass
            self._stop.wait(0.25)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)


def anime_components(alpha: np.ndarray, min_frac: float = 0.004) -> list[tuple[int, int, int, int]]:
    import cv2
    m = (alpha > 0.5).astype(np.uint8)
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    h, w = m.shape
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= min_frac * h * w:
            boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
    return boxes


def click_points(gt: np.ndarray, min_frac: float = 0.004) -> list[tuple[int, int]]:
    """One positive click per reference component: its most interior pixel
    (what a user clicks: the middle of the body, not the outline)."""
    import cv2
    m = (gt > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    h, w = m.shape
    pts = []
    for i in range(1, n):
        if stats[i][4] < min_frac * h * w:
            continue
        comp = (lab == i).astype(np.uint8)
        d = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(d)), d.shape)
        pts.append((int(x), int(y)))
    return pts


# ── propose (reference masks) ──────────────────────────────────────────────
def cmd_propose(args) -> None:
    from backend.models.birefnet import BiRefNetModel
    from backend.models.sam2 import SAM2Model

    root = Path(args.root)
    dev = _device()
    fixes = {}
    if (root / "gt_fix.json").exists():
        fixes = json.loads((root / "gt_fix.json").read_text())
    toon = load_toonout(dev, args.toonout)
    sam = SAM2Model(dev, variant="large", idle_offload_seconds=0)
    only = set(args.only or [])
    for shot in load_shots(root):
        sid = shot["id"]
        if only and sid not in only:
            continue
        fps_ = frame_paths(root, sid)
        rows = []
        for idx in key_frames(len(fps_)):
            img = Image.open(fps_[idx]).convert("RGB")
            fix = fixes.get(sid, {}).get(str(idx)) or fixes.get(sid, {}).get("*", {})
            a_toon = toon.mask_image(img)
            mode = fix.get("mode", "sam")
            if mode == "toon":
                gt = a_toon > 0.5
            elif mode == "skip":  # ambiguous frame (white flash, smear): not scored
                (root / "gt" / sid / f"{idx:05d}.png").unlink(missing_ok=True)
                continue
            elif mode == "empty":
                gt = np.zeros(a_toon.shape, bool)
            else:
                boxes = [tuple(b) for b in fix.get("boxes", [])] or (
                    [] if fix.get("pos") else anime_components(a_toon))
                gt = np.zeros(a_toon.shape, bool)
                pos = [tuple(p) for p in fix.get("pos", [])]
                neg = [tuple(p) for p in fix.get("neg", [])]
                if pos and not fix.get("boxes"):
                    gt |= sam.preview_mask(img, pos, neg)
                for obj in fix.get("objs", []):  # one SAM call per character
                    gt |= sam.preview_mask(img, [tuple(p) for p in obj], neg or None)
                for b in boxes:
                    inside = [p for p in pos if b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]]
                    gt |= sam.preview_mask(img, inside, neg or None, box=b)
                if mode == "sam+toon":
                    # outline detail from the anime matte where SAM agrees on the object
                    import cv2
                    near = cv2.dilate(gt.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                    gt = (gt | ((a_toon > 0.5) & near))
            for (x0, y0, x1, y1) in fix.get("erase", []):
                gt[y0:y1, x0:x1] = False
            for (x0, y0, x1, y1) in fix.get("add", []):
                gt[y0:y1, x0:x1] = True
            gt = clean_reference(gt)
            if fix.get("fill"):
                gt = fill_holes(gt)
            save_alpha(root / "gt" / sid / f"{idx:05d}.png", gt.astype(np.float32))
            rows.append((idx, np.asarray(img), gt))
        review_sheet(root / "review" / f"{sid}.jpg", rows)
        print("ref", sid, flush=True)


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill every background region not connected to the frame border."""
    import cv2
    inv = (~m).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    holes = [i for i in range(1, n) if i not in border]
    out = m.copy()
    if holes:
        out[np.isin(lab, holes)] = True
    return out


def clean_reference(m: np.ndarray, frac: float = 0.0015) -> np.ndarray:
    """Drop specks and fill pinholes smaller than ``frac`` of the frame: SAM
    leaves salt-and-pepper on flat cel-shaded fills that no editor would keep."""
    import cv2
    h, w = m.shape
    lim = frac * h * w
    out = m.copy()
    for target in (True, False):
        n, lab, stats, _ = cv2.connectedComponentsWithStats((out == target).astype(np.uint8), connectivity=8)
        small = np.flatnonzero(stats[:, 4] < lim)
        small = small[small > 0]
        if small.size:
            out[np.isin(lab, small)] = not target
    return out


def _overlay(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    import cv2
    out = rgb.astype(np.float32) * 0.45
    out[m] = rgb[m] * 0.55 + np.array([60, 255, 60]) * 0.45
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[edge] = (255, 0, 255)
    return out.clip(0, 255).astype(np.uint8)


def _grid(img: np.ndarray) -> np.ndarray:
    """10 % grid with coordinate labels, to read click positions off a sheet."""
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    for k in range(1, 10):
        x, y = int(w * k / 10), int(h * k / 10)
        out[:, x] = (out[:, x] * 0.5 + 128).astype(np.uint8)
        out[y, :] = (out[y, :] * 0.5 + 128).astype(np.uint8)
        cv2.putText(out, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        cv2.putText(out, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    return out


def review_sheet(path: Path, rows: list) -> None:
    import cv2
    cells = []
    for idx, rgb, gt in rows:
        a = _grid(rgb)
        b = _overlay(rgb, gt)
        cv2.putText(a, f"#{idx}", (8, a.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cells.append(np.concatenate([a, b], axis=0))
    sheet = np.concatenate(cells, axis=1)
    scale = min(1.0, 3600 / sheet.shape[1])
    sheet = cv2.resize(sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(path, quality=88)


# ── methods ────────────────────────────────────────────────────────────────
def load_toonout(dev, weights: str):
    """BiRefNet architecture (pinned HF code) + the ToonOut fine-tuned weights."""
    from backend.models.birefnet import BiRefNetModel
    m = BiRefNetModel(dev, variant="general", idle_offload_seconds=0)
    m._ensure_loaded()
    import torch
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = m._model.load_state_dict(sd, strict=False)
    if len(missing) > 10 or len(unexpected) > 10:
        raise RuntimeError(f"ToonOut weights do not match BiRefNet: {len(missing)} missing, {len(unexpected)} unexpected")
    m._model.to(m._model_dtype)
    return m


def run_auto(model, paths: list[Path]) -> np.ndarray:
    return np.stack([model.mask_image(Image.open(p).convert("RGB")) for p in paths])


def method_runner(name: str, args, dev):
    """-> fn(paths, gt0, seed_idx) -> (T, H, W) alpha. ``gt0`` is the reference
    of key frame 0, used only to place the simulated user clicks."""
    from backend.models.birefnet import BiRefNetModel
    from backend.models.sam2 import SAM2Model

    if name in ("isnet", "birefnet"):
        m = BiRefNetModel(dev, variant="anime" if name == "isnet" else "general", idle_offload_seconds=0)
        return lambda paths, gt0, k0: run_auto(m, paths)
    if name == "toonout":
        m = load_toonout(dev, args.toonout)
        return lambda paths, gt0, k0: run_auto(m, paths)
    if name.startswith("sam2_") or name.startswith("sam2w_"):
        # sam2_<variant>: the 0.4 code path (whole clip in one SAM 2 state);
        # sam2w_<variant>: 0.5 windowed propagation (bounded memory).
        variant = name.split("_", 1)[1]
        window = 10 ** 6 if name.startswith("sam2_") else 0
        m = SAM2Model(dev, variant=variant, idle_offload_seconds=0)

        def run(paths, gt0, k0):
            pts = click_points(gt0)
            with Image.open(paths[0]) as im:
                size = im.size
            return m.track_keyframes([{"frame_idx": k0, "positive": pts, "negative": [], "box": None}],
                                     frames_dir=paths[0].parent, frame_paths=paths, frame_size=size, soft=True,
                                     window=window)
        return run
    if name.startswith("auto_"):
        # auto_<detector>_<sam variant>_<main|all>[_nofuse]: detector alphas are
        # read back from runs/<detector> (run that method first); its per-frame
        # time is added to this method's time in cmd_run.
        from backend.models.auto_roto import auto_track
        parts = name.split("_")
        det_name, variant, mode = parts[1], parts[2], parts[3]
        nofuse = "nofuse" in parts[4:]
        import backend.models.auto_roto as ar
        ar.CORE_MODE = "adaptive" if "adapt" in parts[4:] else "always"  # 0.5 default: adaptive
        m = SAM2Model(dev, variant=variant, idle_offload_seconds=0)
        root = Path(args.root)

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            det = np.stack([read_alpha(root / "runs" / det_name / sid / f"{i:05d}.png") for i in range(len(paths))])
            small = [np.asarray(Image.open(p).convert("RGB").resize((256, 144))) for p in paths]
            with Image.open(paths[0]) as im:
                size = im.size
            clicks = {k0: (click_points(gt0), [])} if mode == "click" else None
            return auto_track(paths, det, m, frames_dir=paths[0].parent, frame_size=size, mode=mode,
                              frames_small=small, fuse_edges=not nofuse, clicks=clicks)
        run.detector = det_name
        return run
    if name.startswith("ma2_"):
        # MatAnyone 2 (S-Lab, non-commercial) seeded with the auto target of
        # the detector on the clearest frame, propagated both ways.
        from backend.models.auto_roto import select_targets
        det_name, mode = name.split("_")[1], name.split("_")[2]
        root = Path(args.root)
        import torch
        from matanyone2 import InferenceCore, MatAnyone2
        net = MatAnyone2.from_pretrained("PeiqingYang/MatAnyone2").to("cuda").eval()

        def prop(imgs, seed_mask):
            proc = InferenceCore(net, cfg=net.cfg, device="cuda")
            out = []
            # upstream feeds the PNG mask as read (0..255), not 0..1
            mask = torch.from_numpy(seed_mask.astype(np.float32) * 255.0).cuda()
            for ti, im in enumerate(imgs):
                x = torch.from_numpy(im).permute(2, 0, 1).float().cuda() / 255.0
                if ti == 0:
                    proc.step(x, mask, objects=[1])
                    prob = proc.step(x, first_frame_pred=True)
                    for _ in range(9):  # warm-up on the seed frame, as upstream
                        prob = proc.step(x, first_frame_pred=True)
                else:
                    prob = proc.step(x)
                out.append(proc.output_prob_to_mask(prob).float().cpu().numpy())
            return out

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            n = len(paths)
            det = [read_alpha(root / "runs" / det_name / sid / f"{i:05d}.png") for i in range(n)]
            tg = [select_targets(d, mode) for d in det]
            seed = int(np.argmax([t.sum() for t in tg]))
            imgs = [read_rgb(p) for p in paths]
            res = np.zeros((n,) + det[0].shape, np.float32)
            with torch.inference_mode():
                fw = prop(imgs[seed:], tg[seed])
                for j, a in enumerate(fw):
                    res[seed + j] = a
                if seed > 0:
                    bw = prop(imgs[seed::-1], tg[seed])
                    for j, a in enumerate(bw):
                        res[seed - j] = a
            return res
        run.detector = det_name
        return run
    if name in ("bhr", "bhrm"):
        # BiRefNet HR (2048 px input) / HR-matting (soft alpha training)
        m = BiRefNetModel(dev, variant="hr" if name == "bhr" else "hr_matting", idle_offload_seconds=0)
        return lambda paths, gt0, k0: run_auto(m, paths)
    if name.startswith("sam3i-"):
        # SAM 3 image model, one concept prompt per frame: union of the
        # detected instances (score >= 0.4), soft mask probabilities
        prompt = name.split("-", 1)[1].replace("+", " ")
        from backend.models.sam3_concept import Sam3Concept
        m = Sam3Concept(dev)
        return lambda paths, gt0, k0: np.stack([m.segment(read_rgb(p), prompt) for p in paths])
    if name.startswith("sam3z-"):
        # SAM 3 image + a zoomed second pass on each character's crop
        prompt = name.split("-", 1)[1].replace("+", " ")
        from backend.models.sam3_concept import Sam3Concept
        m = Sam3Concept(dev)
        return lambda paths, gt0, k0: np.stack([m.segment_zoomed(read_rgb(p), prompt) for p in paths])
    if name.startswith("union-"):
        # union-<a>,<b>,...: pixel max of several runs (prompt fusion)
        parts = name[len("union-"):].split(",")
        root = Path(args.root)

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            return np.stack([np.max([read_alpha(root / "runs" / q / sid / f"{i:05d}.png") for q in parts], axis=0)
                             for i in range(len(paths))])
        run.detector = parts
        return run
    if name.startswith("sam3v-"):
        # SAM 3 video: concept detection + its own tracker over the shot
        prompt = name.split("-", 1)[1].replace("+", " ")
        from backend.models.sam3_concept import Sam3Concept
        m = Sam3Concept(dev)
        return lambda paths, gt0, k0: m.segment_video([read_rgb(p) for p in paths], prompt)
    if name.startswith("pdet"):
        # deepghs anime person detector (YOLO, MIT): filled boxes as a matte
        from backend.models.anime_person import AnimePersonDetector
        m = AnimePersonDetector()

        def run(paths, gt0, k0):
            out = []
            for p in paths:
                rgb = read_rgb(p)
                a = np.zeros(rgb.shape[:2], np.float32)
                for (x0, y0, x1, y1, _sc) in m.detect(rgb):
                    a[int(y0):int(y1), int(x0):int(x1)] = 1.0
                out.append(a)
            return np.stack(out)
        return run
    if name == "pdsam":
        # deepghs person boxes -> SAM 2.1 large segments inside each box
        from backend.models.anime_person import AnimePersonDetector
        det = AnimePersonDetector()
        sam = SAM2Model(dev, variant="large", idle_offload_seconds=0)

        def run(paths, gt0, k0):
            out = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                rgb = np.asarray(img)
                a = np.zeros(rgb.shape[:2], bool)
                for (x0, y0, x1, y1, _sc) in det.detect(rgb):
                    a |= sam.preview_mask(img, [], None, box=(x0, y0, x1, y1))
                out.append(a.astype(np.float32))
            return np.stack(out)
        return run
    if name.startswith("tgate"):
        # tgate<k>[c<cov>]-<base>-<selector> (or ~-separated when the base
        # name has dashes): tight gate (k thousandths of the diagonal) where
        # the selector covers the base matte, wide gate else
        if "~" in name:
            head, rest = name.split("~", 1)
            base, sel = rest.rsplit("~", 1)
        else:
            head, base, sel = name.split("-", 2)
        k, _, cov = head[5:].partition("c")
        from backend.models.auto_roto import gate_alpha
        root = Path(args.root)

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            out = []
            for i in range(len(paths)):
                b = read_alpha(root / "runs" / base / sid / f"{i:05d}.png")
                g = read_alpha(root / "runs" / sel / sid / f"{i:05d}.png")
                out.append(gate_alpha(b, g, keep_if_empty=True, tight_frac=int(k) / 1000.0,
                                      cover_min=int(cov) / 100.0 if cov else 0.85))
            return np.stack(out)
        run.detector = [base, sel]
        return run
    if name.startswith("hug"):
        # hug<k>[t<touch>][o<opaque>]~<base>~<selector>: tight gate (k thousandths)
        # + opaque tips sticking out of the selector given back
        import re
        head, rest = name.split("~", 1)
        base, sel = rest.rsplit("~", 1)
        mm = re.match(r"hug(\d+)(?:t(\d+))?(?:o(\d+))?(?:r(\d+))?(f)?(h)?", head)
        k, tt, oo, rr, ff, hh = (int(mm.group(1)), mm.group(2), mm.group(3), mm.group(4),
                                 mm.group(5), mm.group(6))
        from backend.models.auto_roto import drop_fragments, hug_with_tips
        root = Path(args.root)

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            out = []
            for i in range(len(paths)):
                b = read_alpha(root / "runs" / base / sid / f"{i:05d}.png")
                g = read_alpha(root / "runs" / sel / sid / f"{i:05d}.png")
                o = hug_with_tips(b, g, tight_frac=k / 1000.0,
                                  max_touch=int(tt) / 100.0 if tt else 0.45,
                                  opaque=int(oo) / 100.0 if oo else 0.9,
                                  reach_frac=int(rr) / 1000.0 if rr else 0.025,
                                  rgb=read_rgb(paths[i]) if hh else None)
                out.append(drop_fragments(o, g) if ff else o)
            return np.stack(out)
        run.detector = [base, sel]
        return run
    if name.startswith("held"):
        # held[p<codes>][k<cover %>][x<max rel x10>][q<score %>][t<touch %%>][l<glow %>][v][r][f]~<base>~<selector>:
        # objects held or worn given back from the BiRefNet matte (runs/birefnet)
        # on SAM 3 object instances (<root>/obj/<shot>/<prompt>/*.png labels +
        # scores.json, written by a multi-prompt SAM 3 pass): auto_roto.held_regions
        # + keep_held. v: temporal persistence (persist_held); r: also give back
        # characters SAM 3 sees that the matte lost (selector components
        # covered < 50 %); l: light-effect filter (max_glow); f: object interior made opaque.
        import re
        head, rest = name.split("~", 1)
        base, sel = rest.rsplit("~", 1)
        codes = {"w": "weapon", "s": "sword", "t": "staff", "h": "halo", "b": "bowl", "m": "hammer",
                 "c": "cup", "g": "wings"}
        mm = re.match(r"held(?:p([wsthbmcg]+))?(?:k(\d+))?(?:x(\d+))?(?:q(\d+))?(?:t(\d+))?(?:l(\d+))?(v)?(r)?(f)?$", head)
        pc, kk, xx, qq, tt, ll, vv, rr, ff = mm.groups()
        prompts = [codes[c] for c in (pc or "wsthbmc")]
        from backend.models import auto_roto as ar
        root = Path(args.root)

        def run(paths, gt0, k0):
            import cv2
            sid = paths[0].parent.parent.name
            od = root / "obj" / sid
            sc = json.loads((od / "scores.json").read_text())
            A, B, H = [], [], []
            for i in range(len(paths)):
                a = read_alpha(root / "runs" / base / sid / f"{i:05d}.png")
                b = read_alpha(root / "runs" / "birefnet" / sid / f"{i:05d}.png")
                g = read_alpha(root / "runs" / sel / sid / f"{i:05d}.png")
                objs = []
                for p in prompts:
                    lab = np.asarray(Image.open(od / p.replace(" ", "+") / f"{i:05d}.png"))
                    for j, s_ in enumerate(sc[p][i]):
                        if s_ >= (int(qq) / 100.0 if qq else 0.4):
                            objs.append(lab == j + 1)
                if rr:
                    n, lab, st, _ = cv2.connectedComponentsWithStats((g > 0.5).astype(np.uint8), connectivity=8)
                    for j in range(1, n):
                        c = lab == j
                        if float((a[c] > 0.5).mean()) < 0.5:
                            objs.append(c)
                hm = ar.held_regions(b, g, objs, min_cover=int(kk) / 100.0 if kk else ar.HELD_MIN_COVER,
                                     rgb=read_rgb(paths[i]) if ll else None,
                                     max_glow=int(ll) / 100.0 if ll else None,
                                     max_rel=int(xx) / 10.0 if xx else ar.HELD_MAX_REL,
                                     touch_frac=int(tt) / 1000.0 if tt else ar.HELD_TOUCH_FRAC)
                A.append(a); B.append(b); H.append(hm)
            if vv:
                H = ar.persist_held(H)
            return np.stack([ar.keep_held(a, b, h, fill_px=2 if ff else None) for a, b, h in zip(A, B, H)])
        run.detector = [base, sel]
        return run
    if name.startswith("sfill"):
        # sfill<pct>-<sel>: motion-blur fill of a selector — on a frame where
        # it sees less than pct % of what it sees on a neighbour (SAM 3 drops
        # smeared limbs), add what both neighbours, warped to this frame by
        # optical flow, agree on; never across a hard cut
        import cv2
        head, sel = name.split("-", 1)
        pct = int(head[5:] or 80) / 100.0
        root = Path(args.root)
        from backend.models.auto_roto import split_shots

        def warp(src_mask, g_to, g_from):
            flow = cv2.calcOpticalFlowFarneback(g_to, g_from, None, 0.5, 4, 21, 3, 5, 1.1, 0)
            h, w = g_to.shape
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            sm = cv2.resize(src_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            return cv2.remap(sm, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR) > 0.5

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            S = np.stack([read_alpha(root / "runs" / sel / sid / f"{i:05d}.png") > 0.5 for i in range(len(paths))])
            rgbs = [read_rgb(p) for p in paths]
            G = [cv2.cvtColor(cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
                 for f in rgbs]
            small = [np.asarray(Image.fromarray(f).resize((256, 144))) for f in rgbs]
            out = S.copy()
            area = S.reshape(len(S), -1).sum(1)
            H, W = S.shape[1:]
            for a, b in split_shots(small, len(paths)):
                for t in range(a + 1, b - 1):
                    if area[t] >= pct * max(area[t - 1], area[t + 1]):
                        continue
                    fill = warp(S[t - 1], G[t], G[t - 1]) & warp(S[t + 1], G[t], G[t + 1])
                    out[t] |= cv2.resize(fill.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            return out.astype(np.float32)
        run.detector = [sel]
        return run
    if name.startswith(("votefill-", "votemaj-")):
        # vote<fill|maj>-<sel>: temporal vote of a selector over t-1, t, t+1
        # (fill: S_t or (S_t-1 and S_t+1); maj: at least 2 of 3); never across
        # a hard cut
        kind, sel = name.split("-", 1)
        root = Path(args.root)
        from backend.models.auto_roto import split_shots

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            S = np.stack([read_alpha(root / "runs" / sel / sid / f"{i:05d}.png") > 0.5 for i in range(len(paths))])
            small = [np.asarray(Image.open(p).convert("RGB").resize((256, 144))) for p in paths]
            out = S.copy()
            for a, b in split_shots(small, len(paths)):
                for t in range(a + 1, b - 1):
                    if kind == "votefill":
                        out[t] = S[t] | (S[t - 1] & S[t + 1])
                    else:
                        out[t] = (S[t - 1].astype(np.uint8) + S[t] + S[t + 1]) >= 2
            return out.astype(np.float32)
        run.detector = [sel]
        return run
    if name.startswith("stab") and not name.startswith("stable"):
        # stab<k>~<base>: the app's temporal stabilisation after tracking
        # (finish.matte.postprocess_stack, strength k/100; the Auto engine
        # uses 0.5 x the setting = 0.25)
        head, base = name.split("~", 1)
        k = int(head[4:]) / 100.0
        root = Path(args.root)
        from backend.finish.matte import postprocess_stack

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            m = np.stack([read_alpha(root / "runs" / base / sid / f"{i:05d}.png") for i in range(len(paths))])
            return postprocess_stack(m, [read_rgb(p) for p in paths], stabilize=True, strength=k)
        run.detector = [base]
        return run
    if name.startswith("avg-"):
        # avg-<a>,<b>,...: mean of several detector mattes (ensemble)
        parts = name[len("avg-"):].split(",")
        root = Path(args.root)

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            return np.stack([np.mean([read_alpha(root / "runs" / q / sid / f"{i:05d}.png") for q in parts], axis=0)
                             for i in range(len(paths))])
        run.detector = parts
        return run
    if name.startswith(("gate-", "gatef-", "gate~", "gatef~")):
        # gate-<base>-<selector> (or gatef~<base>~<selector>): the base matte
        # (edges) kept only where the selector (a detector that knows what a
        # character is) sees one, softly dilated; reads both runs back
        if "~" in name:
            _, base, sel = name.split("~", 2)
        else:
            _, base, sel = name.split("-", 2)
        from backend.models.auto_roto import gate_alpha
        root = Path(args.root)

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            out = []
            for i in range(len(paths)):
                b = read_alpha(root / "runs" / base / sid / f"{i:05d}.png")
                g = read_alpha(root / "runs" / sel / sid / f"{i:05d}.png")
                out.append(gate_alpha(b, g, box=sel.startswith("pdet"), keep_if_empty=name.startswith("gatef")))
            return np.stack(out)
        run.detector = [base, sel]
        return run
    if name.startswith("vm-") or name.startswith("cs-"):
        # vm-<base>: ViTMatte on a trimap around the base matte's outline;
        # cs-<base>: local colour solve of the outline band
        kind, base = name.split("-", 1)
        from backend.models import edge_refine as er
        root = Path(args.root)
        ref = er.EdgeRefiner(dev) if kind == "vm" else None

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            out = []
            for i, p in enumerate(paths):
                a = read_alpha(root / "runs" / base / sid / f"{i:05d}.png")
                rgb = read_rgb(p)
                out.append(ref.refine(rgb, a) if ref else er.colour_solve(rgb, a))
            return np.stack(out)
        run.detector = [base]
        return run
    if name in ("app_stable", "app_max"):
        # the app's Auto chain end to end with the real models (as
        # backend/api/segment.py _track_auto): BiRefNet, SAM 3 (character +
        # held-object concepts, one batched pass; zoomed for max), SAM 2.1
        # tiny tracking ("all"), then stable_masks or tighten, the temporal
        # stabilisation (strength 0.25) and, for stable, the speck cleanup
        from backend.models import auto_roto as ar
        from backend.models.birefnet import BiRefNetModel
        from backend.models.sam3_concept import Sam3Concept
        from backend.finish.matte import postprocess_stack
        bi = BiRefNetModel(dev, variant="general", idle_offload_seconds=0)
        s3 = Sam3Concept(dev)
        sam = SAM2Model(dev, variant="tiny", idle_offload_seconds=0)
        stable = name == "app_stable"

        def run(paths, gt0, k0):
            frames = [read_rgb(p) for p in paths]
            h, w = frames[0].shape[:2]
            det = np.stack([np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)
                            for a in bi.alpha_batch(frames, batch=4)])
            raw = det.copy()
            keep, held = [], []
            gated = ar.apply_selector(det, frames, s3, keep=keep, held=held, zoom=not stable,
                                      reach_frac=ar.STABLE_REACH if stable else ar.GATE_FRAC)
            small = [np.asarray(Image.fromarray(f).resize((256, max(2, round(256 * h / w))))) for f in frames]
            track = ar.auto_track(paths, gated, sam, frames_dir=paths[0].parent, frame_size=(w, h), mode="all",
                                  frames_small=small)
            shots = ar.split_shots(small, len(frames))
            if stable:
                m = ar.stable_masks(raw, keep, track, held, frames=frames, shots=shots)
            else:
                m = ar.tighten(track, keep, quality="max", held=held)
            m = postprocess_stack(m.astype(np.float32), frames, stabilize=True, strength=0.25)
            return ar.clean_masks(m, shots, held) if stable else m
        return run
    if name.startswith("clean~"):
        # clean~<base>: the stable mode's last step, background specks cleaned
        # after the temporal stabilisation (auto_roto.clean_masks; held objects
        # from the held cache of the SAM 3 run used by the base)
        base = name.split("~", 1)[1]
        from backend.models import auto_roto as ar
        root = Path(args.root)

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            m = np.stack([read_alpha(root / "runs" / base / sid / f"{i:05d}.png") for i in range(len(paths))])
            H = None
            for c in sorted((root / "heldcache").glob(f"*/{sid}.npz")):
                z = np.load(c)["h"]
                if len(z) == len(paths):
                    H = [x if x.any() else None for x in z]
                    break
            return ar.clean_masks(m, shot_cuts(root, sid), H)
        run.detector = [base]
        return run
    if name.startswith("stable"):
        # stable[r<reach ‰>][e<erode px>][l<low %>][o<opaque %>][p<soft px>][x<solid reach x10>][k<core px>][j<extra hi %, 0: off>][i<extra lo ‰>][n][t][h][s][c][g]~<track>~<selector>:
        # the "stable" Auto (auto_roto.stable_masks): BiRefNet (runs/birefnet) pieces
        # that touch a character kept, SAM 3 (<selector>) only drops detached ones,
        # the tracked matte (<track>) and held objects (<root>/obj, as heldv) add to
        # it, collapsed frames refilled from their neighbours, specks cleaned.
        # n: no temporal guard; t: no track; h: no held objects; s: SAM 3's mask
        # not added; c: no speck cleanup; p0: no soft-edge limit; g: extras sorted by size only (no colours).
        import re
        head, rest = name.split("~", 1)
        track_name, sel = rest.rsplit("~", 1)
        mm = re.match(r"stable(?:r(\d+))?(?:e(\d+))?(?:l(\d+))?(?:o(\d+))?(?:p(\d+))?(?:x(\d+))?(?:k(\d+))?(?:j(\d+))?(?:i(\d+))?(n)?(t)?(h)?(s)?(c)?(g)?$", head)
        rr, ee, ll, oo, pp, xx, kk, jj, ii, nn, tt, hh, ss, cc, gg = mm.groups()
        from backend.models import auto_roto as ar
        root = Path(args.root)

        def run(paths, gt0, k0):
            sid = paths[0].parent.parent.name
            n = len(paths)
            det = np.stack([read_alpha(root / "runs" / "birefnet" / sid / f"{i:05d}.png") for i in range(n)])
            S = [read_alpha(root / "runs" / sel / sid / f"{i:05d}.png") for i in range(n)]
            tr = None if tt else np.stack([read_alpha(root / "runs" / track_name / sid / f"{i:05d}.png")
                                           for i in range(n)])
            H = None if hh else held_cache(root, sid, paths, det, S, sel)
            kw = dict(reach_frac=int(rr) / 1000.0 if rr else ar.STABLE_REACH,
                      sel_erode=int(ee) if ee else ar.STABLE_SEL_ERODE,
                      low=int(ll) / 100.0 if ll else ar.STABLE_DET_LOW,
                      opaque=int(oo) / 100.0 if oo else ar.STABLE_OPAQUE,
                      soft_px=float(pp) if pp else ar.STABLE_SOFT_PX,
                      solid_reach=int(xx) / 10.0 if xx else ar.STABLE_SOLID_REACH,
                      core_px=float(kk) if kk else ar.STABLE_CORE_PX,
                      extra_hi=(int(jj) / 100.0 if int(jj) else None) if jj else ar.EXTRA_HI,
                      extra_lo=(int(ii) / 1000.0 if ii else min(ar.EXTRA_LO, int(jj) / 100.0) if jj and int(jj)
                                else ar.EXTRA_LO))
            sl = [None] * n if ss else ar.persist_selector(S, shot_cuts(root, sid), frames=paths)
            cuts = shot_cuts(root, sid)
            starts = {a for a, _b in cuts}
            out = np.zeros(det.shape, np.float32)
            for i in range(n):
                out[i] = ar.stable_frame(det[i], sl[i], None if tr is None else tr[i],
                                         None if H is None else H[i], **kw,
                                         prev=None if i in starts else out[i - 1],
                                         rgb=None if gg else read_rgb(paths[i]))
            if not nn:
                out, _fixed = ar.temporal_guard(out, paths, det, cuts, selector=sl)
            if not cc:
                out = ar.clean_masks(out, cuts, H)
            return out
        run.detector = ["birefnet", sel] + ([] if tt else [track_name])
        return run
    raise SystemExit(f"unknown method {name}")


def held_cache(root: Path, sid: str, paths: list, det: np.ndarray, S: list, sel: str) -> list:
    """Per-frame held-object mattes of a shot as the app computes them
    (SAM 3 object instances in <root>/obj, auto_roto.held_regions +
    persist_held + keep_held), cached in <root>/heldcache/<sel>/<shot>.npz."""
    from backend.models import auto_roto as ar
    cache = root / "heldcache" / sel.replace("~", "_") / f"{sid}.npz"
    n = len(paths)
    if cache.exists():
        z = np.load(cache)["h"]
        if len(z) == n:
            return [m if m.any() else None for m in z]
    od = root / "obj" / sid
    if not (od / "scores.json").exists():
        return [None] * n
    sc = json.loads((od / "scores.json").read_text())
    hm = []
    for i in range(n):
        objs = []
        for p in ar.HELD_PROMPTS:
            lab = np.asarray(Image.open(od / p.replace(" ", "+") / f"{i:05d}.png"))
            objs += [lab == j + 1 for j, s_ in enumerate(sc[p][i]) if s_ >= 0.4]
        hm.append(ar.held_regions(det[i], S[i], objs, rgb=read_rgb(paths[i])))
    H = []
    for i, m in enumerate(ar.persist_held(hm)):
        if m is None or not np.any(m):
            H.append(np.zeros(det.shape[1:], np.uint8))
        else:
            H.append(np.clip(np.rint(ar.keep_held(np.zeros(m.shape, np.float32), det[i], m) * 255), 0, 255)
                     .astype(np.uint8))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, h=np.stack(H))
    return [m if m.any() else None for m in H]


def shot_cuts(root: Path, sid: str) -> list[tuple[int, int]]:
    """[start, end) ranges between the hard cuts of a shot (the app's cut
    detection, auto_roto.split_shots), cached in <root>/cuts.json."""
    cache = root / "cuts.json"
    allc = json.loads(cache.read_text()) if cache.exists() else {}
    if sid not in allc:
        from backend.models.auto_roto import split_shots
        paths = frame_paths(root, sid)
        small = [np.asarray(Image.open(p).convert("RGB").resize((256, 144))) for p in paths]
        allc[sid] = [list(r) for r in split_shots(small, len(paths))]
        cache.write_text(json.dumps(allc))
    return [tuple(r) for r in allc[sid]]


def cmd_run(args) -> None:
    import torch
    root = Path(args.root)
    dev = _device()
    fn = method_runner(args.method, args, dev)
    shots = load_shots(root)
    only = set(args.only or [])
    warm = False
    with _VramPoll() as vram:
        for shot in shots:
            sid = shot["id"]
            if only and sid not in only:
                continue
            if not args.redo and (root / "runs" / args.method / sid / "stats.json").exists():
                continue  # resume: already computed
            paths = frame_paths(root, sid)
            k0 = seed_frame(root, sid, len(paths))
            gt0 = read_alpha(root / "gt" / sid / f"{k0:05d}.png")
            if not warm:  # lazy init + kernel selection, on a 3-frame copy of the shot
                import shutil
                import tempfile
                wd = Path(tempfile.mkdtemp(prefix="roto_warm_")) / sid / "frames"
                wd.mkdir(parents=True)
                for p in paths[:3]:
                    shutil.copy(p, wd / p.name)
                if getattr(fn, "detector", None) and isinstance(fn.detector, str):
                    for i in range(3):
                        shutil.copy(root / "runs" / fn.detector / sid / f"{i:05d}.png",
                                    wd / f"det{i:05d}.png")
                try:
                    fn(sorted(wd.glob("0*.png")), gt0, 0)
                except Exception as exc:
                    print("warm-up failed (ignored):", exc, flush=True)
                shutil.rmtree(wd.parent.parent, ignore_errors=True)
                warm = True
            if dev.backend == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            alpha = fn(paths, gt0, k0)
            if dev.backend == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            out = root / "runs" / args.method / sid
            for i, a in enumerate(alpha):
                save_alpha(out / f"{i:05d}.png", a)
            ms = 1000 * dt / len(paths)
            det_name = getattr(fn, "detector", None)
            for d_ in ([det_name] if isinstance(det_name, str) else (det_name or [])):
                # the detector passes this method reads back are part of its cost
                ms += json.loads((root / "runs" / d_ / sid / "stats.json").read_text())["ms_per_frame"]
            stats = {"ms_per_frame": ms, "n": len(paths),
                     "vram_mb": vram.peak_mb,
                     "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
            (out / "stats.json").write_text(json.dumps(stats))
            print(args.method, sid, json.dumps(stats), flush=True)


# ── score ──────────────────────────────────────────────────────────────────
def _boundary(m: np.ndarray) -> np.ndarray:
    import cv2
    m8 = m.astype(np.uint8)
    return (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0


def f_boundary(pred: np.ndarray, gt: np.ndarray) -> float:
    import cv2
    h, w = gt.shape
    tol = max(1, int(round(0.008 * np.hypot(h, w))))
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() and not bg.any():
        return 1.0
    if not bp.any() or not bg.any():
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gd = cv2.dilate(bg.astype(np.uint8), k) > 0
    pd = cv2.dilate(bp.astype(np.uint8), k) > 0
    prec = (bp & gd).sum() / bp.sum()
    rec = (bg & pd).sum() / bg.sum()
    return float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0


def edge_error(alpha: np.ndarray, gt: np.ndarray, band: int = 8) -> float:
    import cv2
    bg = _boundary(gt)
    if not bg.any():
        return float("nan")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    zone = cv2.dilate(bg.astype(np.uint8), k) > 0
    return float(np.abs(alpha[zone] - gt[zone].astype(np.float32)).mean() * 100)


FLOW_SCALE = 0.5


def shot_flows(root: Path, sid: str, idxs: list[int]) -> dict:
    """Backward flows (t -> t-1) at half resolution for consecutive pairs of
    ``idxs``, cached per shot: they do not depend on the method scored.
    Pairs the flow cannot explain (cuts, very fast action: mean warped
    photometric error > 12/255) are marked invalid — there the change is
    legitimate and the metric would only measure the flow's failure."""
    import cv2
    cache = root / "flow" / f"{sid}.npz"
    if cache.exists():
        d = np.load(cache)
        if list(d["idxs"]) == list(idxs):
            return {"idxs": idxs, "maps": d["maps"], "valid": d["valid"]}
    paths = frame_paths(root, sid)
    maps, valid, prev_g = [], [], None
    for t in idxs:
        f = read_rgb(paths[t])
        g = cv2.cvtColor(cv2.resize(f, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY)
        if prev_g is not None:
            flow = cv2.calcOpticalFlowFarneback(g, prev_g, None, 0.5, 4, 21, 3, 5, 1.1, 0)
            h, w = g.shape
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            mx, my = gx + flow[..., 0], gy + flow[..., 1]
            warped = cv2.remap(prev_g, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            valid.append(np.abs(warped.astype(np.float32) - g).mean() <= 12)
            maps.append(np.stack([mx, my]).astype(np.float16))
        prev_g = g
    maps = np.stack(maps) if maps else np.zeros((0, 2, 1, 1), np.float16)
    valid = np.array(valid, bool)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, idxs=np.array(idxs), maps=maps, valid=valid)
    return {"idxs": idxs, "maps": maps, "valid": valid}


def flicker(alphas: list[np.ndarray], flows: dict) -> float:
    """Flow-compensated temporal change: sum |a_t - warp(a_t-1)| / sum a_t (%)."""
    import cv2
    num = den = 0.0
    small = [cv2.resize(a, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA) for a in alphas]
    for k in range(1, len(small)):
        if not flows["valid"][k - 1]:
            continue
        mx, my = flows["maps"][k - 1].astype(np.float32)
        wa = cv2.remap(small[k - 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        num += float(np.abs(small[k] - wa).sum())
        den += float(small[k].sum())
    return 100.0 * num / den if den > 0 else 0.0


BAND_PX = 8          # outline band (hair strands, halo) at the 1080 px working size
REF_TOL_PX = 2       # the SAM 2.1 reference outline is only right to ~2 px: not scored
ENVELOPE_PX = 41     # closing that turns the silhouette into its envelope


def bg_zones(gt: np.ndarray):
    """(band_out, band_in, pockets, d_out, d_in) of a binary reference."""
    import cv2
    g = gt.astype(np.uint8)
    d_out = cv2.distanceTransform(1 - g, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(g, cv2.DIST_L2, 3)
    band_out = (g == 0) & (d_out > REF_TOL_PX) & (d_out <= BAND_PX)
    band_in = (g == 1) & (d_in > REF_TOL_PX) & (d_in <= BAND_PX)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ENVELOPE_PX, ENVELOPE_PX))
    env = fill_holes(cv2.morphologyEx(g, cv2.MORPH_CLOSE, k) > 0)
    pockets = env & (g == 0) & (d_out > BAND_PX)
    return band_out, band_in, pockets, d_out, d_in


def bg_metrics(a: np.ndarray, gt: np.ndarray) -> dict:
    band_out, band_in, pockets, _, _ = bg_zones(gt)
    out = {"bg_band": float(a[band_out].mean() * 100) if band_out.sum() >= 50 else float("nan"),
           "eaten": float((1 - a[band_in]).mean() * 100) if band_in.sum() >= 50 else float("nan"),
           "bg_in": float(a[pockets].mean() * 100) if pockets.sum() >= 200 else float("nan")}
    return out


def held_kept(a: np.ndarray, ref: Path) -> float:
    """Held object kept (%): mean exported alpha on the reference mask of the
    objects the characters hold or wear (<root>/gt_held, a subset of the
    reference), its outer REF_TOL_PX px not scored; nan without one."""
    import cv2
    if not ref.exists():
        return float("nan")
    m = (read_alpha(ref) > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * REF_TOL_PX + 1, 2 * REF_TOL_PX + 1))
    m = cv2.erode(m, k) > 0
    return float(a[m].mean() * 100) if m.sum() >= 30 else float("nan")


def contamination(rgb_out: np.ndarray, rgb_src: np.ndarray, a: np.ndarray, gt: np.ndarray) -> float:
    """Background colour left in the semi-transparent edge pixels of an
    export (%). For each such pixel, F = source colour of the nearest
    reference-character pixel 3 px or more inside the outline (the ink line
    when there is one), B = mean source colour of the reference background
    around it, away from the outline; the exported colour C scores
    c = (C - F).(B - F) / |B - F|^2 in [0, 1] — 0: character colour,
    1: background colour — weighted by its alpha. Only pixels whose F and B
    differ (|B - F| >= 40/255) count: elsewhere spill is invisible."""
    import cv2
    g = gt > 0.5
    _, _, _, d_out, d_in = bg_zones(g)
    src = rgb_src.astype(np.float32) / 255.0
    col = rgb_out.astype(np.float32) / 255.0
    core = g & (d_in >= 3)
    if not core.any():
        return float("nan")
    _, lab = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 3,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    cy, cx = np.nonzero(core)
    idx = np.clip(lab.astype(np.int64) - 1, 0, len(cy) - 1)
    F = src[cy[idx], cx[idx]]
    bgm = (~g & (d_out > 8)).astype(np.float32)
    wb = cv2.blur(bgm, (21, 21))
    B = cv2.blur(src * bgm[..., None], (21, 21)) / (wb[..., None] + 1e-6)
    near = (d_out <= 8) | (d_in <= 8)
    fb = B - F
    den = (fb * fb).sum(axis=2)
    ok = near & (a > 0.05) & (a < 0.95) & (wb > 0.02) & (den >= (40 / 255) ** 2)
    if ok.sum() < 30:
        return float("nan")
    c = np.clip(((col - F) * fb).sum(axis=2) / np.maximum(den, 1e-6), 0, 1)
    return float((a[ok] * c[ok]).sum() / a[ok].sum() * 100)


def synthetic_composite(rgb: np.ndarray, gt: np.ndarray, sigma: float = 0.8):
    """Ground-truth composite of a key frame: (I, F, B, alpha).

    F: the character's colours (reference pixels as they are; outside the
    reference, the colour of the nearest reference pixel), B: the frame's own
    background (colour of the nearest background pixel), alpha: the reference
    softened into a ~2 px antialiased ramp. I = alpha F + (1 - alpha) B. On
    I the true foreground colour of every pixel is known, so contamination
    can be measured exactly instead of guessing F from the frame (the old
    estimate counted correct ink pixels as contamination: 26 % on a perfect
    answer, see docs/reports/chantier-roto.md, part 4)."""
    import cv2
    g = gt > 0.5
    if not g.any():
        return None
    _, lab = cv2.distanceTransformWithLabels((~g).astype(np.uint8), cv2.DIST_L2, 3,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    gy, gx = np.nonzero(g)
    k = np.clip(lab.astype(np.int64) - 1, 0, len(gy) - 1)
    F = rgb[gy[k], gx[k]]
    # background: colour of the nearest real background pixel 4 px or more
    # outside the reference (known exactly, fast; only the outline band needs it)
    far = cv2.dilate(g.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    if not far.any():
        return None
    _, labb = cv2.distanceTransformWithLabels((~far).astype(np.uint8), cv2.DIST_L2, 3,
                                              labelType=cv2.DIST_LABEL_PIXEL)
    by, bx = np.nonzero(far)
    kb = np.clip(labb.astype(np.int64) - 1, 0, len(by) - 1)
    B = rgb[by[kb], bx[kb]]
    alpha = cv2.GaussianBlur(g.astype(np.float32), (0, 0), sigma)
    I = (alpha[..., None] * F + (1 - alpha[..., None]) * B).round().clip(0, 255).astype(np.uint8)
    return I, F, B, alpha


def contamination_true(col: np.ndarray, a_out: np.ndarray, F: np.ndarray, B: np.ndarray,
                       alpha: np.ndarray) -> tuple[float, float]:
    """(contamination %, colour error) of an export of a synthetic composite:
    background share of the exported colour on the true F -> B axis,
    weighted by the exported alpha, on the pixels the truth calls
    semi-transparent where F and B differ visibly; colour error = mean
    |C - F| (0..255) there, which also catches over-correction."""
    c = col.astype(np.float32) / 255.0
    f = F.astype(np.float32) / 255.0
    b = B.astype(np.float32) / 255.0
    fb = b - f
    den = (fb * fb).sum(2)
    ok = (alpha > 0.05) & (alpha < 0.95) & (a_out > 0.05) & (den >= (40 / 255) ** 2)
    if ok.sum() < 30:
        return float("nan"), float("nan")
    t = np.clip(((c - f) * fb).sum(2) / np.maximum(den, 1e-6), 0, 1)
    w = a_out[ok]
    cont = float((w * t[ok]).sum() / w.sum() * 100)
    err = float(np.sqrt(((c - f) ** 2).sum(2))[ok].mean() * 255 / np.sqrt(3))
    return cont, err


def export_post(name: str, rgb: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What the alpha export does to (frame, alpha) at the working size.
    ``none``: alpha as is, source colours; ``v05``: the 0.5 export (guided
    filter on the alpha + colour-only background-aware edge cleanup);
    ``v06``: the 0.6 export (matte as is + foreground estimation of the
    edge colours, backend/models/mask_postprocess.export_edges); ``fg:...``,
    ``ce``, ``kb``: variants measured while choosing it."""
    from backend.models import mask_postprocess as mp
    if name == "none":
        return rgb, a
    if name == "v05":
        a2 = mp.upsample_alpha_guided(a, rgb)
        return mp.clean_edges_with_background(rgb, a2, resolve_alpha=False)
    if name == "v06":
        return mp.export_edges(rgb, a)
    if name.startswith("um"):  # um[c]<faint>_<solid>: unmix against an inpainted background (c: after colour solve)
        from backend.models import edge_refine as er
        body = name[2:]
        cs = body.startswith("c")
        body = body[1:] if cs else body
        fa, so = (int(x) / 100.0 for x in body.split("_")) if body else (0.15, 0.5)
        a2 = er.colour_solve_safe(rgb, a, band_frac=0.012, in_frac=0.002) if cs else a
        return er.unmix_with_background(rgb, a2, faint=fa, solid=so), a2
    if name == "max":  # the max-quality export of the app (edge_refine.finish_edges)
        from backend.models.edge_refine import finish_edges
        return finish_edges(rgb, a, "max")
    if name == "fg":  # foreground estimation only, alpha untouched
        from backend.models.edge_refine import estimate_foreground
        return estimate_foreground(rgb, a), a
    if name.startswith("fg:"):  # fg:r1:r2[:u]
        from backend.models.edge_refine import estimate_foreground
        ps = name[3:].split(":")
        nb = [float(x[1:]) for x in ps[2:] if x.startswith("n")]
        return estimate_foreground(rgb, a, int(ps[0]), int(ps[1]), unmix="u" in ps,
                                   nearest_below=nb[0] if nb else 0.0), a
    if name in ("cs", "cs2"):  # colour-solved alpha (cs2: ink protected) + foreground estimation
        from backend.models import edge_refine as er
        a2 = er.colour_solve(rgb, a) if name == "cs" else er.colour_solve_safe(rgb, a)
        return er.estimate_foreground(rgb, a2), a2
    if name.startswith("csu"):  # csu<in>_<out>: csi alpha, colours unmixed against the nearest background
        from backend.models import edge_refine as er
        from backend.models import mask_postprocess as mp
        ri, ro = (int(x) / 1000.0 for x in name[3:].split("_"))
        a2 = er.colour_solve_safe(rgb, a, band_frac=ro, in_frac=ri)
        col, _ = mp.clean_edges_with_background(rgb, a2, resolve_alpha=False)
        return col, a2
    if name.startswith("csb"):  # csb<in>_<out>: csi alpha, then unmix, then foreground estimation
        from backend.models import edge_refine as er
        from backend.models import mask_postprocess as mp
        ri, ro = (int(x) / 1000.0 for x in name[3:].split("_"))
        a2 = er.colour_solve_safe(rgb, a, band_frac=ro, in_frac=ri)
        col, _ = mp.clean_edges_with_background(rgb, a2, resolve_alpha=False)
        return er.estimate_foreground(col, a2), a2
    if name.startswith("csi"):  # csi<in>_<out>[_<ink%>][n<near%>]: asymmetric band, ink lines protected
        from backend.models import edge_refine as er
        near = 0.0
        if "n" in name:
            name, _, nn = name.partition("n")
            near = int(nn) / 100.0
        ps = name[3:].split("_")
        a2 = er.colour_solve_safe(rgb, a, band_frac=int(ps[1]) / 1000.0, in_frac=int(ps[0]) / 1000.0,
                                  ink_margin=int(ps[2]) / 100.0 if len(ps) > 2 else 0.12)
        return er.estimate_foreground(rgb, a2, nearest_below=near), a2
    if name.startswith("csa"):  # csa<in>_<out> (thousandths of the diagonal): asymmetric band
        from backend.models import edge_refine as er
        ri, ro = (int(x) / 1000.0 for x in name[3:].split("_"))
        a2 = er.colour_solve(rgb, a, band_frac=ro, in_frac=ri)
        return er.estimate_foreground(rgb, a2), a2
    if name.startswith("lv"):  # lv<lo>_<hi> (percent): alpha levels + foreground estimation
        from backend.models import edge_refine as er
        lo, hi = (int(x) / 100.0 for x in name[2:].split("_"))
        a2 = er.alpha_levels(a, lo, hi)
        return er.estimate_foreground(rgb, a2), a2
    if name == "ce":  # 0.5 colour cleanup without the guided filter
        return mp.clean_edges_with_background(rgb, a, resolve_alpha=False)
    if name == "kb":  # background kill + foreground estimation
        from backend.models.edge_refine import estimate_foreground, kill_background
        a2 = kill_background(rgb, a)
        return estimate_foreground(rgb, a2), a2
    raise SystemExit(f"unknown export {name}")


def score_method(root: Path, method: str, shots: list[dict], export: str = "v05") -> dict:
    res = {}
    for shot in shots:
        sid = shot["id"]
        rd = root / "runs" / method / sid
        if not (rd / "stats.json").exists() or shot.get("excluded"):
            continue  # excluded: reference found wrong during the visual review
        paths = frame_paths(root, sid)
        js, fs, es, fails = [], [], [], 0
        bgs = {"bg_band": [], "bg_in": [], "eaten": [], "contam": [], "contam_true": [], "colerr": [], "contam_col": [],
               "held": []}
        for idx in scored_frames(root, sid, len(paths)):
            gt = read_alpha(root / "gt" / sid / f"{idx:05d}.png") > 0.5
            a = read_alpha(rd / f"{idx:05d}.png")
            rgb = read_rgb(paths[idx])
            rgb_e, a_e = export_post(export, rgb, a)
            for k_, v_ in bg_metrics(a_e, gt).items():
                bgs[k_].append(v_)
            bgs["held"].append(held_kept(a_e, root / "gt_held" / sid / f"{idx:05d}.png"))
            bgs["contam"].append(contamination(rgb_e, rgb, a_e, gt))
            # ground-truth contamination: the same export run on a composite
            # of this frame whose true colours are known (method's alpha)
            syn = None if os.environ.get("ROTO_NO_CONTAM") else synthetic_composite(rgb, gt)
            if syn is not None:
                I_s, F_s, B_s, al_s = syn
                col_s, a_s = export_post(export, I_s, a)
                ct, ce = contamination_true(col_s, a_s, F_s, B_s, al_s)
                bgs["contam_true"].append(ct)
                bgs["colerr"].append(ce)
                # same, with the true alpha given to the export: the colour step alone
                col_p, a_p = export_post(export, I_s, al_s)
                cp, _cpe = contamination_true(col_p, a_p, F_s, B_s, al_s)
                bgs["contam_col"].append(cp)
            p = a > 0.5
            union = (p | gt).sum()
            j = float((p & gt).sum() / union) if union else 1.0
            js.append(j)
            fs.append(f_boundary(p, gt))
            e = edge_error(a, gt)
            if not np.isnan(e):
                es.append(e)
            fails += j < 0.5
        sub = list(range(min(len(paths), 96)))  # flicker on up to 95 consecutive pairs
        alphas = [read_alpha(rd / f"{i:05d}.png") for i in sub]
        # stability on EVERY frame, no reference needed (roto stable)
        stab = stability(root, sid, rd, len(paths))
        st = json.loads((rd / "stats.json").read_text())
        res[sid] = {"J": float(np.mean(js)), "F": float(np.mean(fs)),
                    "edge": float(np.mean(es)) if es else float("nan"),
                    "flicker": flicker(alphas, shot_flows(root, sid, sub)), "fail": fails, "n_scored": len(js),
                    **stab,
                    **{k_: float(np.nanmean(v_)) if np.isfinite(v_).any() else float("nan")
                       for k_, v_ in ((k, np.array(v, float)) for k, v in bgs.items())},
                    **st}
    return res


def stability(root: Path, sid: str, rd: Path, n: int) -> dict:
    """Stability of a method's mattes on every frame of a shot, no reference
    needed: ``lost`` frames (matte area < 60 % of the rolling median of the
    shot, +-12 frames: the character vanished or was eaten), ``jumps``
    (area change > 25 % between neighbouring frames), ``lost_ids`` /
    ``jump_ids`` (frame numbers, for the visual check), and ``held_seq`` /
    ``held_min``: the held object kept on every frame that has a held
    reference in <root>/gt_held_seq (mean / worst %)."""
    from backend.models.auto_roto import area_jumps, lost_frames, specks
    a = np.stack([read_alpha(rd / f"{i:05d}.png") for i in range(n)])
    cuts = shot_cuts(root, sid)
    lost = lost_frames(a, cuts)
    jumps = area_jumps(a, cuts)
    sp = np.array([specks(x) for x in a], float)
    out = {"lost": len(lost), "jumps": len(jumps), "lost_ids": lost, "jump_ids": jumps, "n_frames": n,
           "area_med": float(np.median([(x > 0.5).mean() * 100 for x in a])),
           "islands": float(sp[:, 0].mean()), "orphans": float(sp[:, 1].mean())}
    hs = []
    hd = root / "gt_held_seq" / sid
    if hd.exists():
        for p in sorted(hd.glob("*.png")):
            i = int(p.stem)
            if i < n:
                hs.append(held_kept(a[i], p))
    hs = [x for x in hs if np.isfinite(x)]
    out["held_seq"] = float(np.mean(hs)) if hs else float("nan")
    out["held_min"] = float(np.min(hs)) if hs else float("nan")
    out["held_seq_n"] = len(hs)
    out["held_seq_lost"] = int(sum(x < 50 for x in hs))
    return out


def cmd_score(args) -> None:
    root = Path(args.root)
    shots = load_shots(root)
    methods = args.methods or sorted(p.name for p in (root / "runs").iterdir() if p.is_dir())
    out = Path(args.scores) if args.scores else root / "scores.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    for m in methods:
        # "<method>@<export>": the same run scored after another export path
        name, _, export = m.partition("@")
        prev[m] = score_method(root, name, shots, export or args.export)
        print("scored", m, flush=True)
    out.write_text(json.dumps(prev, indent=1))
    print(table(prev, shots))
    print()
    print(stable_table(prev, shots))


def table(scores: dict, shots: list[dict]) -> str:
    cats = []
    for s in shots:
        if s["cat"] not in cats:
            cats.append(s["cat"])
    lines = ["| méthode | catégorie | J | F | J&F | bord | scint. % | échecs | fond contour % | fond intérieur % "
             "| mangé % | contam. % | objet tenu % | ms/img | VRAM Mo |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, per in scores.items():
        for cat in cats + ["TOUT"]:
            ids = [s["id"] for s in shots if (cat == "TOUT" or s["cat"] == cat) and s["id"] in per]
            if not ids:
                continue
            g = lambda k: float(np.nanmean([per[i].get(k, np.nan) for i in ids]))  # noqa: E731
            fails = sum(per[i]["fail"] for i in ids)
            nsc = sum(per[i]["n_scored"] for i in ids)
            vram = max(per[i]["vram_mb"] for i in ids)
            lines.append(f"| {m} | {cat} | {g('J'):.3f} | {g('F'):.3f} | {(g('J') + g('F')) / 2:.3f} | "
                         f"{g('edge'):.1f} | {g('flicker'):.1f} | {fails}/{nsc} | {g('bg_band'):.1f} | {g('bg_in'):.1f} | "
                         f"{g('eaten'):.1f} | {g('contam'):.1f} | {g('held'):.1f} | {g('ms_per_frame'):.0f} | {vram} |")
    return "\n".join(lines)


def stable_table(scores: dict, shots: list[dict]) -> str:
    """Roto stable table: stability on every frame first (lost frames,
    jumps), then the outline error with the eaten character weighing as
    much as the background kept (``contour`` = fond contour + mangé)."""
    cats = []
    for s in shots:
        if s["cat"] not in cats:
            cats.append(s["cat"])
    lines = ["| méthode | catégorie | images | perdues | sauts > 25 % | scint. % | J | fond contour % | (médiane) | mangé % "
             "| contour (fond + mangé) | fond intérieur % | objet tenu % | objet tenu, toutes images % "
             "| images sans l'objet | îlots / image | px isolés / image |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, per in scores.items():
        for cat in cats + ["TOUT"]:
            ids = [s["id"] for s in shots if (cat == "TOUT" or s["cat"] == cat) and s["id"] in per
                   and "lost" in per[s["id"]]]
            if not ids:
                continue
            g = lambda k: float(np.nanmean([per[i].get(k, np.nan) for i in ids]))  # noqa: E731
            tot = lambda k: int(sum(per[i].get(k, 0) for i in ids))  # noqa: E731
            med = lambda k: float(np.nanmedian([per[i].get(k, np.nan) for i in ids]))  # noqa: E731
            lines.append(f"| {m} | {cat} | {tot('n_frames')} | {tot('lost')} | {tot('jumps')} | {g('flicker'):.1f} | "
                         f"{g('J'):.3f} | {g('bg_band'):.1f} | {med('bg_band'):.1f} | {g('eaten'):.1f} | {g('bg_band') + g('eaten'):.1f} | "
                         f"{g('bg_in'):.1f} | {g('held'):.1f} | {g('held_seq'):.1f} | "
                         f"{tot('held_seq_lost')}/{tot('held_seq_n')} | {g('islands'):.2f} | {g('orphans'):.0f} |")
    return "\n".join(lines)


# ── boards ─────────────────────────────────────────────────────────────────
def _checker(h, w, s=16):
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // s) + (xx // s)) % 2).astype(np.float32)
    return (c[..., None] * 60 + 160).repeat(3, axis=2)


def composite(rgb: np.ndarray, a: np.ndarray, bg: np.ndarray) -> np.ndarray:
    return (rgb * a[..., None] + bg * (1 - a[..., None])).clip(0, 255).astype(np.uint8)


def cmd_boards(args) -> None:
    import cv2
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = args.methods
    for shot in load_shots(root):
        sid = shot["id"]
        paths = frame_paths(root, sid)
        idxs = scored_frames(root, sid, len(paths))
        idxs = [idxs[0], idxs[len(idxs) // 2], idxs[-1]]
        rows = []
        for m in ["ref"] + methods:
            cells = []
            for idx in idxs:
                rgb = read_rgb(paths[idx]).astype(np.float32)
                ap = (root / "gt" / sid / f"{idx:05d}.png") if m == "ref" else (root / "runs" / m / sid / f"{idx:05d}.png")
                if not ap.exists():
                    a = np.zeros(rgb.shape[:2], np.float32)
                else:
                    a = read_alpha(ap)
                h, w = a.shape
                c1 = composite(rgb, a, _checker(h, w))
                c2 = composite(rgb, a, np.full((h, w, 3), (40, 200, 60), np.float32))
                cell = np.concatenate([c1, c2], axis=1)
                cv2.putText(cell, f"{m} #{idx}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
                cv2.putText(cell, f"{m} #{idx}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
                cells.append(cell)
            rows.append(np.concatenate(cells, axis=1))
        board = np.concatenate(rows, axis=0)
        scale = min(1.0, args.width / board.shape[1])
        board = cv2.resize(board, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        Image.fromarray(board).save(out_dir / f"{sid}.jpg", quality=88)
        print("board", sid, flush=True)


def hair_window(gt: np.ndarray, size: tuple[int, int] = (240, 150)) -> tuple[int, int, int, int]:
    """Crop (x0, y0, x1, y1) where the reference outline is the most
    convoluted — hair spikes, strands, fingers — favouring the upper half."""
    import cv2
    h, w = gt.shape
    cw, ch = min(size[0], w), min(size[1], h)
    b = _boundary(gt > 0.5).astype(np.float32)
    if not b.any():
        return 0, 0, cw, ch
    dens = cv2.boxFilter(b, -1, (cw, ch), normalize=False)
    dens *= np.linspace(1.3, 0.7, h, dtype=np.float32)[:, None]
    y, x = np.unravel_index(int(np.argmax(dens)), dens.shape)
    x0 = int(np.clip(x - cw // 2, 0, w - cw))
    y0 = int(np.clip(y - ch // 2, 0, h - ch))
    return x0, y0, x0 + cw, y0 + ch


def cmd_zoom(args) -> None:
    """Hair close-ups: per shot, 3 scored key frames cropped where the outline
    is the most detailed, zoomed x3 (nearest), exported matte composited on
    magenta (any source-background colour left in the matte stands out).
    Rows: frame, reference, then each ``method[@export]``."""
    import cv2
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only or [])
    for shot in load_shots(root):
        sid = shot["id"]
        if (only and sid not in only) or shot.get("excluded"):
            continue
        paths = frame_paths(root, sid)
        idxs = scored_frames(root, sid, len(paths))
        if not idxs:
            continue
        idxs = [idxs[0], idxs[len(idxs) // 2], idxs[-1]]
        rows = []
        for m in ["frame", "ref"] + list(args.methods or []):
            cells = []
            for idx in idxs:
                gt = read_alpha(root / "gt" / sid / f"{idx:05d}.png")
                x0, y0, x1, y1 = hair_window(gt)
                rgb = read_rgb(paths[idx])
                if m == "frame":
                    cell = rgb
                else:
                    name, _, export = m.partition("@")
                    if m == "ref":
                        a, col = gt, rgb
                    else:
                        ap = root / "runs" / name / sid / f"{idx:05d}.png"
                        a = read_alpha(ap) if ap.exists() else np.zeros(gt.shape, np.float32)
                        col, a = export_post(export or "v05", rgb, a)
                    mag = np.zeros_like(rgb, dtype=np.float32)
                    mag[...] = {"magenta": (255, 0, 255), "black": (0, 0, 0), "white": (255, 255, 255),
                                "green": (0, 200, 0)}[getattr(args, "bg", "magenta")]
                    cell = composite(col.astype(np.float32), a, mag)
                    if m == "ref":  # reference outline in cyan
                        e = _boundary(gt > 0.5)
                        cell = cell.copy()
                        cell[e] = (0, 255, 255)
                crop = cv2.resize(np.ascontiguousarray(cell[y0:y1, x0:x1]), None, fx=args.scale, fy=args.scale,
                                  interpolation=cv2.INTER_NEAREST)
                cv2.putText(crop, f"{m} #{idx}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
                cv2.putText(crop, f"{m} #{idx}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cells.append(crop)
            rows.append(np.concatenate(cells, axis=1))
        board = np.concatenate(rows, axis=0)
        Image.fromarray(board).save(out_dir / f"{sid}.jpg", quality=90)
        print("zoom", sid, flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "propose", "run", "score", "boards", "zoom"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--only", nargs="*")
        if name == "extract":
            p.add_argument("--src", nargs="+", required=True)
            p.add_argument("--max-dim", type=int, default=1080)
        if name in ("propose", "run"):
            p.add_argument("--toonout", default=os.environ.get("TOONOUT_WEIGHTS", ""))
        if name == "run":
            p.add_argument("--method", required=True)
            p.add_argument("--redo", action="store_true", help="recompute shots already done")
        if name in ("score", "boards", "zoom"):
            p.add_argument("--methods", nargs="*")
        if name == "score":
            p.add_argument("--export", default="v05", help="none | v05 | v06 (default for methods without @)")
            p.add_argument("--scores", help="scores file (default <root>/scores.json); one per process "
                                            "to score methods in parallel")
        if name in ("boards", "zoom"):
            p.add_argument("--out", required=True)
            p.add_argument("--width", type=int, default=2400)
        if name == "zoom":
            p.add_argument("--bg", default="magenta", choices=["magenta", "black", "white", "green"])
            p.add_argument("--scale", type=int, default=3)
    args = ap.parse_args(argv)
    {"extract": cmd_extract, "propose": cmd_propose, "run": cmd_run,
     "score": cmd_score, "boards": cmd_boards, "zoom": cmd_zoom}[args.cmd](args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
