"""FFmpeg-based export and proxy generation."""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ..indexing.cuts import _ffmpeg_path, probe_video

log = logging.getLogger(__name__)

_PROXY_BITRATES = {"low": "300k", "medium": "500k", "high": "1000k"}


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# ── NVENC probe ─────────────────────────────────────────────────────────────
_NVENC_PROBED = False
_NVENC_AVAILABLE = False


def _has_nvenc_h264() -> bool:
    """True if this ffmpeg has h264_nvenc compiled in. Cached per process."""
    global _NVENC_PROBED, _NVENC_AVAILABLE
    if _NVENC_PROBED:
        return _NVENC_AVAILABLE
    _NVENC_PROBED = True
    try:
        proc = subprocess.run(
            [_ffmpeg_path(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
            creationflags=_creation_flags(),
        )
        _NVENC_AVAILABLE = "h264_nvenc" in (proc.stdout or "")
        if _NVENC_AVAILABLE:
            log.info("h264_nvenc available — proxies will be GPU-encoded")
        else:
            log.info("h264_nvenc not available — proxies will use libx264 CPU (slower)")
    except Exception as exc:
        log.warning("NVENC probe failed: %s", exc)
    return _NVENC_AVAILABLE


def export_scene(
    src_path: str,
    start_ms: int,
    end_ms: int,
    output_path: str,
    *,
    codec: str = "libx264",
    crf: int = 18,
    audio: str = "copy",
    resolution: str = "source",
) -> str:
    """Frame-accurate export with two-pass seek (fast then exact)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    start_sec = start_ms / 1000.0
    duration_sec = max(0.05, (end_ms - start_ms) / 1000.0)
    buffer_sec = 10.0
    fast_seek = max(0.0, start_sec - buffer_sec)
    exact_seek = start_sec - fast_seek

    # Map the setting name to the actual ffmpeg encoder. Notable case:
    # `dnxhr` (what the user picks) is produced by ffmpeg's `dnxhd` encoder —
    # the dnxhr_* profile flag is what switches it to the modern variant.
    encoder = "dnxhd" if codec == "dnxhr" else codec

    cmd = [
        _ffmpeg_path(),
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{fast_seek}",
        "-i", src_path,
        "-ss", f"{exact_seek}",
        "-t", f"{duration_sec}",
        "-c:v", encoder,
    ]
    if codec in ("libx264", "libx265", "libsvtav1", "libvpx-vp9"):
        cmd.extend(["-crf", str(crf), "-preset", "medium"])
    elif codec == "prores_ks":
        # ProRes profile 3 = ProRes 422 HQ, the sane default for delivery.
        cmd.extend(["-profile:v", "3"])
    elif codec == "dnxhr":
        # DNxHR is a parametric codec — ffmpeg refuses to encode without an
        # explicit profile. `dnxhr_sq` is the 4:2:2 8-bit standard variant,
        # the most universal choice for editorial workflows.
        cmd.extend(["-profile:v", "dnxhr_sq", "-pix_fmt", "yuv422p"])

    if resolution != "source":
        height = {"1080p": 1080, "720p": 720, "480p": 480}.get(resolution)
        if height:
            cmd.extend(["-vf", f"scale=-2:{height}"])

    if audio == "copy":
        cmd.extend(["-c:a", "copy"])
    elif audio == "encode":
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    elif audio == "mute":
        cmd.extend(["-an"])

    cmd.extend(["-map", "0:v:0", "-map", "0:a?", "-avoid_negative_ts", "make_zero", "-y", output_path])

    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=_creation_flags())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr}")
    return output_path


def generate_proxy(src_path: str, start_ms: int, end_ms: int, output_path: str,
                   *, quality: str = "medium") -> str:
    """Generate a small H.264 .mp4 (or VP9 .webm fallback) proxy for hover preview.

    Optimised for the use case: a silent, low-fps, low-res clip that loads
    instantly when the user mouses over a scene. We don't need audio (browser
    plays it muted on hover anyway), we don't need 30 fps (humans don't notice
    on a 320×180 thumb), and we don't need full vertical resolution.

    With h264_nvenc on a consumer NVIDIA card: ~30-50× realtime → ~5-10 s of
    encode for the full ~300 proxies of a 22-min episode. With the libx264
    fallback: ~10-15 s per proxy = several minutes per episode.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bitrate = _PROXY_BITRATES.get(quality, _PROXY_BITRATES["medium"])
    start_sec = start_ms / 1000.0
    duration_sec = max(0.05, (end_ms - start_ms) / 1000.0)

    use_nvenc = _has_nvenc_h264() and output_path.lower().endswith(".mp4")

    base = [
        _ffmpeg_path(),
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0, start_sec - 0.5)}",
        "-i", src_path,
        "-ss", f"{min(start_sec, 0.5)}",
        "-t", f"{duration_sec}",
    ]

    if use_nvenc:
        # `p1` = fastest preset, `tune ll` = low-latency, ~30-50× realtime on
        # consumer cards. Muted, 240p, 15 fps is way enough for hover previews.
        cmd = base + [
            "-c:v", "h264_nvenc",
            "-preset", "p1",
            "-tune", "ll",
            "-b:v", bitrate,
            "-vf", "scale=-2:240,fps=15",
            "-an",
            "-y", output_path,
        ]
    elif output_path.lower().endswith(".webm"):
        # Legacy VP9 path — kept so existing .webm files in the DB can still be
        # regenerated if a user explicitly asks for it.
        cmd = base + [
            "-c:v", "libvpx-vp9",
            "-b:v", bitrate,
            "-vf", "scale=-2:360,fps=24",
            "-c:a", "libopus", "-b:a", "48k", "-ac", "1",
            "-row-mt", "1", "-cpu-used", "5",
            "-y", output_path,
        ]
    else:
        # CPU H.264 fallback for boxes without NVENC.
        cmd = base + [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-b:v", bitrate,
            "-vf", "scale=-2:240,fps=15",
            "-an",
            "-y", output_path,
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=_creation_flags())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg proxy failed: {proc.stderr}")
    return output_path


def _write_mask_png_seq(masks: np.ndarray, out_dir: Path) -> None:
    """Write a ``(T, H, W)`` mask stack as ``00000.png``…``NNNNN.png`` 8-bit
    grayscale PNGs that ffmpeg's ``alphamerge`` reads as the alpha channel.

    Accepts either bool (binary mask) or float32 [0, 1] (continuous alpha).
    Per-frame post-processing (shrink, BG suppression) is handled by the
    caller before passing the masks in — keeps this function trivial."""
    from PIL import Image

    for i, m in enumerate(masks):
        if m.dtype == bool:
            arr = (m.astype(np.uint8) * 255)
        else:
            arr = (np.clip(m.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(arr, "L").save(out_dir / f"{i:05d}.png", optimize=False)


def _source_video_size(src_path: str) -> tuple[int, int] | None:
    try:
        resolution = str(probe_video(src_path).get("resolution") or "")
        w_s, h_s = resolution.lower().split("x", 1)
        w, h = int(w_s), int(h_s)
        if w > 0 and h > 0:
            return w, h
    except Exception as exc:
        log.warning("could not probe source size for alpha export: %s", exc)
    return None


def _encoder_args_for_alpha(codec: str) -> list[str]:
    if codec == "prores_4444_alpha":
        return ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]
    if codec == "vp9_alpha":
        return [
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0",
            "-b:v", "2M",
            "-row-mt", "1",
        ]
    raise ValueError(f"unknown alpha codec: {codec}")


def _resize_alpha(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)
    img = Image.fromarray((alpha * 65535.0).astype(np.uint16), "I;16")
    resized = img.resize(size, Image.BILINEAR)
    return (np.asarray(resized).astype(np.float32) / 65535.0).clip(0.0, 1.0)


def _export_soft_alpha_as_rgba_sequence(
    src_path: str,
    start_sec: float,
    duration_sec: float,
    masks: np.ndarray,
    output_path: str,
    *,
    codec: str,
    fps: float,
) -> str:
    """Export soft alpha as decontaminated RGBA frames.

    A plain alphamerge keeps original RGB under semi-transparent pixels, so the
    old scene background bleeds into hair edges when composited elsewhere.
    This path replaces soft-edge RGB with nearby confident-foreground colour
    before encoding ProRes/WebM alpha.
    """
    from PIL import Image
    from ..models.mask_postprocess import decontaminate_rgb_with_alpha

    with tempfile.TemporaryDirectory(prefix="amv_soft_alpha_") as tmp:
        tmp_path = Path(tmp)
        rgb_dir = tmp_path / "rgb"
        rgba_dir = tmp_path / "rgba"
        rgb_dir.mkdir()
        rgba_dir.mkdir()

        extract_cmd = [
            _ffmpeg_path(),
            "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_sec}",
            "-t", f"{duration_sec}",
            "-i", src_path,
            "-vf", f"fps={fps}",
            "-frames:v", str(masks.shape[0]),
            "-start_number", "0",
            str(rgb_dir / "%05d.png"),
        ]
        proc = subprocess.run(extract_cmd, capture_output=True, text=True, creationflags=_creation_flags())
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg soft-alpha frame extract failed: {proc.stderr[-800:]}")

        frame_paths = sorted(rgb_dir.glob("*.png"))
        if not frame_paths:
            raise RuntimeError("ffmpeg produced no RGB frames for soft-alpha export")

        n = min(len(frame_paths), masks.shape[0])
        for i in range(n):
            with Image.open(frame_paths[i]) as im:
                rgb = np.asarray(im.convert("RGB"))
            alpha = _resize_alpha(masks[i], (rgb.shape[1], rgb.shape[0]))
            clean_rgb = decontaminate_rgb_with_alpha(rgb, alpha)
            rgba = np.dstack([clean_rgb, (alpha * 255.0).round().astype(np.uint8)])
            Image.fromarray(rgba, "RGBA").save(rgba_dir / f"{i:05d}.png", optimize=False)

        cmd = [
            _ffmpeg_path(),
            "-hide_banner", "-loglevel", "error",
            "-framerate", f"{fps}",
            "-i", str(rgba_dir / "%05d.png"),
            "-an",
            *_encoder_args_for_alpha(codec),
            "-y", output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=_creation_flags())
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg soft-alpha export failed: {proc.stderr[-800:]}")
    return output_path


def export_scene_with_alpha(
    src_path: str,
    start_ms: int,
    end_ms: int,
    masks: np.ndarray,
    output_path: str,
    *,
    codec: str = "prores_4444_alpha",
    fps: float = 24.0,
    decontaminate_rgb: bool = False,
) -> str:
    """Encode the clip with the SAM 2 masks as the alpha channel.

    ``codec`` is either ``prores_4444_alpha`` (ProRes 4444, 10-bit, ``.mov`` —
    Resolve/AE/Premiere all consume it natively) or ``vp9_alpha`` (VP9 + alpha
    in ``.webm`` — lighter, web-friendly, less universally supported).

    Pipeline:
        Input 0 = source clip with ``-ss/-t`` trim (keeps native fps and res).
        Input 1 = PNG sequence of masks at the same fps as the SAM 2 extract,
                  sized to whatever ``predict_video_masks`` returned.
        filter  = scale mask to source res, then ``alphamerge``.

    The mask res may differ from the source res (we resize during extraction
    to keep SAM 2 fast). ffmpeg upscales the mask with bilinear inside the
    filter chain — usable for AMV compositing, but if you need pixel-perfect
    you'd want to re-run SAM 2 at native resolution.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    start_sec = max(0.0, start_ms / 1000.0)
    duration_sec = max(0.05, (end_ms - start_ms) / 1000.0)

    if masks.ndim != 3:
        raise ValueError(f"masks must be (T, H, W), got shape {masks.shape}")

    if decontaminate_rgb and np.issubdtype(masks.dtype, np.floating):
        return _export_soft_alpha_as_rgba_sequence(
            src_path,
            start_sec,
            duration_sec,
            masks,
            output_path,
            codec=codec,
            fps=fps,
        )

    with tempfile.TemporaryDirectory(prefix="amv_alpha_") as tmp:
        tmp_path = Path(tmp)
        _write_mask_png_seq(masks, tmp_path)

        if codec == "prores_4444_alpha":
            # Profile 4 = ProRes 4444 (10-bit + alpha). yuva444p10le is the
            # pixel format that actually carries the alpha channel — without
            # it ffmpeg would silently drop the alpha plane.
            encoder = ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]
        elif codec == "vp9_alpha":
            # libvpx-vp9 + yuva420p = VP9 with embedded alpha in WebM. The
            # `auto-alt-ref` flag must be off when emitting alpha, otherwise
            # libvpx silently strips it.
            encoder = [
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuva420p",
                "-auto-alt-ref", "0",
                "-b:v", "2M",
                "-row-mt", "1",
            ]
        else:
            raise ValueError(f"unknown alpha codec: {codec}")

        # alphamerge needs a grayscale mask at the same resolution as the
        # source. Prefer an explicit source-sized scale because scale2ref can
        # crash on some Windows ffmpeg builds.
        source_size = _source_video_size(src_path)
        if source_size is not None:
            source_w, source_h = source_size
            filter_complex = (
                f"[1:v]scale={source_w}:{source_h}:flags=bilinear,format=gray[maskg];"
                "[0:v][maskg]alphamerge[out]"
            )
        else:
            filter_complex = (
                "[1:v][0:v]scale2ref=flags=bilinear[mask][base];"
                "[mask]format=gray[maskg];"
                "[base][maskg]alphamerge[out]"
            )

        cmd = [
            _ffmpeg_path(),
            "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_sec}",
            "-t", f"{duration_sec}",
            "-i", src_path,
            "-framerate", f"{fps}",
            "-i", str(tmp_path / "%05d.png"),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-an",  # alpha exports for compositing don't carry audio
            *encoder,
            "-y", output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=_creation_flags())
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg alpha export failed: {proc.stderr[-800:]}")
    return output_path
