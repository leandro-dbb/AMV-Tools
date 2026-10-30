"""Export of a roto project, for editing software and for AMV Tools itself.

One pass over the shot at the source's full resolution writes, as asked:

- ``<name>_alpha.mov``   ProRes 4444 with alpha (After Effects, Premiere,
                         Resolve, Vegas), clean background, edge colours
                         decontaminated (the Stable v5 export);
- ``<name>_png/``        the same frames as an RGBA PNG sequence;
- ``<name>_matte.mov``   the matte alone, black and white (ProRes 422 HQ).

The edge settings are applied on the working-size matte first (same code as
the preview, :mod:`backend.roto.edges`), then the matte is scaled up onto
the full-resolution drawing (guided filter) and the ink line added there.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from PIL import Image

from .edges import apply_edges

log = logging.getLogger(__name__)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "shot"


def final_alpha_work(project, t: int) -> np.ndarray:
    """Working-size matte of frame ``t`` (all layers) with the edge settings
    except the ink line (added at full resolution)."""
    with project.lock:
        a = project.combined_u8(t).astype(np.float32) / 255.0
        s = project.settings
        prev = nxt = None
        if s.temporal > 0:
            if t > 0:
                prev = project.combined_u8(t - 1).astype(np.float32) / 255.0
            if t + 1 < project.n:
                nxt = project.combined_u8(t + 1).astype(np.float32) / 255.0
    return apply_edges(a, project.rgb(t), s, prev=prev, nxt=nxt,
                       rgb_prev=project.rgb(t - 1) if prev is not None else None,
                       rgb_next=project.rgb(t + 1) if nxt is not None else None, ink=False)


def finish_full(rgb: np.ndarray, alpha_work: np.ndarray, keep_ink: bool) -> tuple[np.ndarray, np.ndarray]:
    """(rgb uint8, alpha float) at the resolution of ``rgb``."""
    from ..models.mask_postprocess import upsample_alpha_guided
    a = alpha_work
    if a.shape != rgb.shape[:2]:
        a = upsample_alpha_guided(a, rgb)
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    if keep_ink:
        from ..models.edge_refine import finish_edges
        return finish_edges(rgb, a, "soft")
    from ..models.edge_refine import MAX_UNMIX_FAINT, MAX_UNMIX_SOLID, unmix_with_background
    return unmix_with_background(rgb, a, faint=MAX_UNMIX_FAINT, solid=MAX_UNMIX_SOLID), a


def _ffmpeg(args: list) -> None:
    from ..export.ffmpeg import _creation_flags, _ffmpeg_path
    proc = subprocess.run([_ffmpeg_path(), "-hide_banner", "-loglevel", "error", *args],
                          capture_output=True, text=True, creationflags=_creation_flags())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")


def export_project(project, out_dir: Path, formats: Sequence[str], *,
                   progress: Callable[[int, int], None] = lambda a, b: None,
                   cancel: Callable[[], None] = lambda: None,
                   name: Optional[str] = None) -> Dict[str, str]:
    """Write the requested ``formats`` ("prores", "png", "matte") and
    return ``{format: path}``."""
    from ..export.ffmpeg import _encoder_args_for_alpha, extract_png_frames
    formats = [f for f in formats if f in ("prores", "png", "matte")] or ["prores"]
    base = _sanitize(name or project.meta.get("name") or project.id)
    dest = Path(out_dir) / f"{base}_roto"
    dest.mkdir(parents=True, exist_ok=True)
    fps = float(project.meta["fps"])
    n = project.n
    files: Dict[str, str] = {}
    tmp = Path(tempfile.mkdtemp(prefix="amv_roto_export_"))
    try:
        video = project.meta.get("video_path")
        if video and Path(video).exists():
            full = extract_png_frames(video, int(project.meta["start_ms"]), int(project.meta["end_ms"]),
                                      tmp / "src_%05d.png", fps=fps, max_frames=n)
            if abs(len(full) - n) > 1:
                log.warning("roto export: %d source frames for %d mattes", len(full), n)
        else:
            full = list(project.frame_paths)   # no source (benchmark): the working frames
        n_out = min(n, len(full))
        png_dir = dest / f"{base}_png" if "png" in formats else tmp / "rgba"
        if png_dir.exists():
            shutil.rmtree(png_dir)
        png_dir.mkdir(parents=True)
        matte_dir = tmp / "matte"
        matte_dir.mkdir()
        keep_ink = bool(project.settings.keep_ink)
        for i in range(n_out):
            cancel()
            with Image.open(full[i]) as im:
                rgb = np.asarray(im.convert("RGB"))
            rgb_o, a = finish_full(rgb, final_alpha_work(project, i), keep_ink)
            a8 = (np.clip(a, 0, 1) * 255.0).round().astype(np.uint8)
            Image.fromarray(np.dstack([rgb_o, a8]), "RGBA").save(png_dir / f"{i:05d}.png", compress_level=1)
            if "matte" in formats:
                Image.fromarray(a8, "L").save(matte_dir / f"{i:05d}.png", compress_level=1)
            progress(i + 1, n_out + 1)
        if "prores" in formats:
            mov = dest / f"{base}_alpha.mov"
            _ffmpeg(["-framerate", f"{fps}", "-i", str(png_dir / "%05d.png"), "-an",
                     *_encoder_args_for_alpha("prores_4444_alpha"), "-y", str(mov)])
            files["prores"] = str(mov)
        if "matte" in formats:
            mov = dest / f"{base}_matte.mov"
            _ffmpeg(["-framerate", f"{fps}", "-i", str(matte_dir / "%05d.png"), "-an",
                     "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-y", str(mov)])
            files["matte"] = str(mov)
        if "png" in formats:
            files["png_dir"] = str(png_dir)
        progress(n_out + 1, n_out + 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return files
