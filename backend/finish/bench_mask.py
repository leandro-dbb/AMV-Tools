"""Mask Mode before/after benchmark on synthetic clips with known alpha.

    python -m backend.finish.bench_mask --models-dir ~/.cache/amv-tools/mask-models

Needs ``isnetis.onnx`` (skytnt/anime-seg, Auto engine "anime" variant) and,
for the Manual engine, ``sam2.1_hiera_tiny.pt`` in ``--models-dir``. Every
pipeline sees the same model outputs; only the pre/post-processing differs.

"before" reproduces the 0.3.0 code paths exactly:
  - Auto: per-frame ISNet -> ``temporal_smooth_alpha(strength=0.5)`` -> export
    cleanup (``clean_edges_with_background``).
  - Manual: one click + box on frame 0, forward-only propagation, boolean
    masks -> export cleanup.
"after":
  - Auto: per-frame ISNet -> flow-compensated gated stabilisation -> colour
    guided-filter edge refinement -> same export cleanup.
  - Manual: same click/box, bidirectional soft (sigmoid) propagation -> same
    post-processing. "+kf" adds one extra click keyframe (first frame of the
    second half / first frame after a cut); "+fix" additionally feeds the
    worst frame's corrected matte back as a mask keyframe (simulates a brush
    correction followed by re-propagation).
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("bench_mask")


def _patch_hf(models_dir: Path) -> None:
    import huggingface_hub

    orig = huggingface_hub.hf_hub_download
    local = {p.name: str(p) for p in models_dir.iterdir() if p.is_file()}

    def fake(repo_id, filename, *a, **kw):
        if filename in local:
            return local[filename]
        return orig(repo_id, filename, *a, **kw)

    huggingface_hub.hf_hub_download = fake


def _export_stage(frames, masks, *, legacy: bool):
    """Export-time edge cleanup (default ON in settings). ``legacy`` = the
    0.3 alpha re-solve; otherwise the 0.4 colour-only cleanup.
    Returns (rgb stack, alpha stack)."""
    from ..models.mask_postprocess import clean_edges_with_background
    rgb = np.empty_like(frames)
    out = np.empty(masks.shape, dtype=np.float32)
    for i, (f, m) in enumerate(zip(frames, masks)):
        rgb[i], out[i] = clean_edges_with_background(f, m.astype(np.float32), resolve_alpha=legacy)
    return rgb, out


def _prompt_from_gt(gt: np.ndarray) -> dict:
    ys, xs = np.nonzero(gt > 0.5)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    # nudge the click onto the silhouette if the median falls outside
    if gt[cy, cx] <= 0.5:
        k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
        cy, cx = int(ys[k]), int(xs[k])
    h, w = gt.shape
    box = (max(0, xs.min() - 4), max(0, ys.min() - 4), min(w - 1, xs.max() + 4), min(h - 1, ys.max() + 4))
    return {"positive": [(cx, cy)], "negative": [], "box": tuple(int(v) for v in box)}


def _cached(cache_dir: Path | None, key: str, fn):
    """Model outputs are expensive on CPU (~10 s/frame for ISNet here);
    cache them so post-processing changes can be re-measured quickly."""
    if cache_dir is None:
        return fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}.npy"
    if p.exists():
        return np.load(p)
    out = fn()
    np.save(p, out)
    return out


def run(models_dir: Path, scenarios: list[str], n_frames: int, w: int, h: int, with_sam2: bool,
        sprite_path: Path | None = None, cache_dir: Path | None = None, roundtrip: bool = False) -> dict:
    from PIL import Image

    from ..models.birefnet import BiRefNetModel
    from ..models.device import detect_device
    from ..models.mask_postprocess import temporal_smooth_alpha
    from .matte import postprocess_stack
    from .synthetic import codec_roundtrip, composite_error, load_sprite, matte_metrics, render_clip

    _patch_hf(models_dir)
    device = detect_device("cpu")
    isnet = BiRefNetModel(device, variant="anime", idle_offload_seconds=0)
    sam2 = None
    if with_sam2:
        from ..models.sam2 import SAM2Model
        sam2 = SAM2Model(device, variant="tiny", idle_offload_seconds=0)

    sprite = load_sprite(str(sprite_path)) if sprite_path else None
    tag = f"{'sprite' if sprite is not None else 'geo'}_{n_frames}_{w}x{h}"
    results: dict = {}
    for sc in scenarios:
        frames, gt, fg = render_clip(sc, n_frames=n_frames, w=w, h=h, sprite=sprite, return_fg=True)
        if roundtrip:
            frames = codec_roundtrip(frames)
        res: dict = {}

        def score(rgb_alpha):
            rgb, alpha = rgb_alpha
            m = matte_metrics(alpha, gt)
            m["comp"] = composite_error(rgb, alpha, gt, fg)
            return m

        t0 = time.time()
        raw = _cached(cache_dir, f"isnet_{sc}_{tag}{'_rt' if roundtrip else ''}",
                      lambda: np.stack([isnet.mask_image(Image.fromarray(f)) for f in frames]))
        t_model = time.time() - t0
        res["auto_raw"] = score((frames, raw))
        res["auto_before"] = score(_export_stage(frames, temporal_smooth_alpha(raw, strength=0.5), legacy=True))
        t1 = time.time()
        res["auto_after"] = score(_export_stage(frames, postprocess_stack(raw, frames, strength=0.5), legacy=False))
        res["auto_post_s_per_frame"] = (time.time() - t1) / n_frames
        res["auto_model_s_per_frame"] = t_model / n_frames

        if sam2 is not None:
            with tempfile.TemporaryDirectory() as tmp:
                tmpd = Path(tmp)
                paths = []
                for i, f in enumerate(frames):
                    p = tmpd / f"{i:05d}.png"
                    Image.fromarray(f).save(p)
                    paths.append(p)
                kw = dict(frames_dir=tmpd, frame_paths=paths, frame_size=(w, h))
                kf0 = {"frame_idx": 0, **_prompt_from_gt(gt[0])}
                hard = _cached(cache_dir, f"sam2_hard_{sc}_{tag}",
                               lambda: sam2.track_keyframes([kf0], soft=False, **kw))
                res["sam2_before"] = score(_export_stage(frames, hard.astype(np.float32), legacy=True))
                soft = _cached(cache_dir, f"sam2_soft_{sc}_{tag}",
                               lambda: sam2.track_keyframes([kf0], soft=True, **kw))
                res["sam2_after"] = score(_export_stage(frames, postprocess_stack(soft, frames, strength=0.25),
                                                        legacy=False))
                mid = n_frames // 2
                from ..finish.synthetic import SCENARIOS
                cut_at = SCENARIOS[sc].cut_at
                if cut_at is not None:
                    mid = cut_at if cut_at < n_frames else n_frames // 2
                kf1 = {"frame_idx": mid, **_prompt_from_gt(gt[mid])}
                soft_kf = _cached(cache_dir, f"sam2_kf_{sc}_{tag}",
                                  lambda: sam2.track_keyframes([kf0, kf1], soft=True, **kw))
                pp_kf = postprocess_stack(soft_kf, frames, strength=0.25)
                res["sam2_after_kf"] = score(_export_stage(frames, pp_kf, legacy=False))
                ious = [((p > 0.5) & (g > 0.5)).sum() / max(1, ((p > 0.5) | (g > 0.5)).sum())
                        for p, g in zip(pp_kf, gt)]
                worst = int(np.argmin(ious))
                kf2 = {"frame_idx": worst, "mask": gt[worst] > 0.5}
                soft_fix = _cached(cache_dir, f"sam2_fix_{sc}_{tag}_{worst}",
                                   lambda: sam2.track_keyframes([kf0, kf1, kf2], soft=True, **kw))
                res["sam2_after_kf_fix"] = score(
                    _export_stage(frames, postprocess_stack(soft_fix, frames, strength=0.25), legacy=False))
                res["sam2_fix_frame"] = worst
        results[sc] = res
        log.info("%s: %s", sc, json.dumps(res, indent=None))
    return results


def to_markdown(results: dict) -> str:
    rows = ["| clip | pipeline | IoU ↑ | IoU min ↑ | edge err ↓ | flicker ↓ | composite err ↓ |",
            "|---|---|---|---|---|---|---|"]
    for sc, res in results.items():
        for key, m in res.items():
            if not isinstance(m, dict):
                continue
            rows.append(f"| {sc} | {key} | {m['iou']:.3f} | {m['iou_min']:.3f} | {m['edge_mae']:.2f} | "
                        f"{m['flicker']:.2f} | {m.get('comp', float('nan')):.2f} |")
    return "\n".join(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--scenarios", default="walk,fast,lowcontrast,cut")
    ap.add_argument("--no-sam2", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--sprite", type=Path, help="RGBA character cutout (else a geometric figure)")
    ap.add_argument("--cache-dir", type=Path, help="cache raw model outputs here")
    ap.add_argument("--roundtrip", action="store_true", help="H.264-compress the clip first (realistic)")
    args = ap.parse_args()
    res = run(args.models_dir.expanduser(), args.scenarios.split(","), args.frames,
              args.width, args.height, not args.no_sam2,
              sprite_path=args.sprite.expanduser() if args.sprite else None,
              cache_dir=args.cache_dir.expanduser() if args.cache_dir else None,
              roundtrip=args.roundtrip)
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
