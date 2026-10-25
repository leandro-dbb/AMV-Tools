"""FFmpeg-based export and proxy generation."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..indexing.cuts import _ffmpeg_path

log = logging.getLogger(__name__)

_PROXY_BITRATES = {"low": "300k", "medium": "500k", "high": "1000k"}

# The "adaptive hardware H.264" codec family — both remap transparently to
# whichever hardware encoder the machine actually has (NVENC on NVIDIA,
# VideoToolbox on Apple Silicon), libx264 otherwise.
_HW_H264_CODECS = ("h264_nvenc", "h264_videotoolbox")


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# ── hardware encoder probe ──────────────────────────────────────────────────
_ENCODERS_PROBED = False
_ENCODERS: set[str] = set()


def _available_encoders() -> set[str]:
    """Names of the encoders compiled into this ffmpeg. Cached per process."""
    global _ENCODERS_PROBED, _ENCODERS
    if _ENCODERS_PROBED:
        return _ENCODERS
    _ENCODERS_PROBED = True
    try:
        proc = subprocess.run(
            [_ffmpeg_path(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
            creationflags=_creation_flags(),
        )
        _ENCODERS = {
            line.split()[1]
            for line in (proc.stdout or "").splitlines()
            if len(line.split()) >= 2 and line.lstrip()[:1] in ("V", "A", "S")
        }
    except Exception as exc:
        log.warning("encoder probe failed: %s", exc)
    return _ENCODERS


def _has_nvenc_h264() -> bool:
    return "h264_nvenc" in _available_encoders()


def _has_videotoolbox_h264() -> bool:
    # The encoder is compiled into every macOS build; actual hardware encode
    # exists on all Apple Silicon (and most Intel) Macs. `-allow_sw 1` keeps
    # it working even on the rare machine without the hardware block.
    return sys.platform == "darwin" and "h264_videotoolbox" in _available_encoders()


def _videotoolbox_decode_args(src_path: str) -> list[str]:
    """``-hwaccel videotoolbox`` only where it beats software decode."""
    from ..indexing.cuts import _decode_profile
    try:
        prof = _decode_profile(src_path)
    except Exception:
        prof = None
    return ["-hwaccel", "videotoolbox"] if prof and prof["name"] == "videotoolbox" else []


_HW_ENCODER_LOGGED = False


def _hw_h264_encoder() -> Optional[str]:
    """Best hardware H.264 encoder on this machine, or None (libx264 only)."""
    global _HW_ENCODER_LOGGED
    enc = None
    if _has_nvenc_h264():
        enc = "h264_nvenc"
    elif _has_videotoolbox_h264():
        enc = "h264_videotoolbox"
    if not _HW_ENCODER_LOGGED:
        _HW_ENCODER_LOGGED = True
        if enc:
            log.info("%s available — proxies/exports will be GPU-encoded", enc)
        else:
            log.info("no hardware H.264 encoder — falling back to libx264 CPU (slower)")
    return enc


# Bits-per-pixel of Adobe Media Encoder's "Match Source — Adaptive High
# Bitrate" H.264 preset: it targets 15.2 Mbps for 1080p23.976, i.e.
# 15.2e6 / (1920*1080*23.976) ≈ 0.306 bit/pixel. We reuse that constant so
# the NVENC preset produces the same target at any source resolution/fps.
_ADAPTIVE_BPP = 0.306
_ADAPTIVE_MIN_BPS = 4_000_000
_ADAPTIVE_MAX_BPS = 40_000_000


def _adaptive_bitrate(src_path: str, resolution: str) -> int:
    """AME-style adaptive VBR target from the output's pixel rate."""
    from ..indexing.cuts import probe_video

    w, h, fps = 1920, 1080, 24.0
    try:
        info = probe_video(src_path)
        fps = float(info.get("fps") or 24.0)
        res = str(info.get("resolution") or "")
        w_s, h_s = res.lower().split("x", 1)
        w, h = int(w_s) or 1920, int(h_s) or 1080
    except Exception as exc:
        log.warning("adaptive bitrate probe failed (%s) — assuming 1080p24", exc)
    out_h = {"1080p": 1080, "720p": 720, "480p": 480}.get(resolution)
    if out_h and h > 0:
        w = int(round(w * out_h / h))
        h = out_h
    target = int(_ADAPTIVE_BPP * w * h * min(120.0, max(1.0, fps)))
    return max(_ADAPTIVE_MIN_BPS, min(_ADAPTIVE_MAX_BPS, target))


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
    audio_bitrate_kbps: int = 320,
    _gpu_decode: bool = True,
) -> str:
    """Frame-accurate export with two-pass seek (fast then exact)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    start_sec = start_ms / 1000.0
    duration_sec = max(0.05, (end_ms - start_ms) / 1000.0)
    # Pre-roll decoded before the clip and thrown away. ffmpeg's input seek
    # is frame-accurate on its own: on Apple Silicon, 12 clips of an H.264
    # and an HEVC episode decode to the same frames (framemd5) without it,
    # 41 % faster. Other platforms keep it until checked there.
    buffer_sec = 0.0 if sys.platform == "darwin" else 10.0
    fast_seek = max(0.0, start_sec - buffer_sec)
    exact_seek = start_sec - fast_seek

    # Map the setting name to the actual ffmpeg encoder. Notable cases:
    # `dnxhr` (what the user picks) is produced by ffmpeg's `dnxhd` encoder —
    # the dnxhr_* profile flag is what switches it to the modern variant.
    # `h264_nvenc`/`h264_videotoolbox` (the "adaptive hardware H.264" family)
    # silently remap to whatever hardware encoder this machine actually has,
    # and fall back to libx264 (same VBR targets) when there is none — so a
    # library exported on an NVIDIA box re-exports fine on a Mac and
    # vice-versa.
    if codec == "auto":
        # Hardware H.264 when the machine has it (NVENC: 3-10x faster than
        # libx264 on a 6-core VM, higher fidelity at its adaptive bitrate),
        # else the libx264 CRF export users had before.
        codec = _hw_h264_encoder() or "libx264"
    encoder = "dnxhd" if codec == "dnxhr" else codec
    if codec in _HW_H264_CODECS:
        requested_ok = (_has_nvenc_h264() if codec == "h264_nvenc"
                        else _has_videotoolbox_h264())
        if not requested_ok:
            encoder = _hw_h264_encoder() or "libx264"
            log.warning("%s requested but unavailable — using %s (same VBR targets)", codec, encoder)

    # NVENC: decode with NVDEC and keep the frames on the GPU (scale_cuda
    # converts 10-bit sources to the NV12 that h264_nvenc takes). The
    # software decode of the pre-roll + clip was most of an export.
    gpu_frames = _gpu_decode and encoder == "h264_nvenc" and _cuda_decode_available()
    cmd = [
        _ffmpeg_path(),
        "-hide_banner", "-loglevel", "error",
        *(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] if gpu_frames else []),
        "-ss", f"{fast_seek}",
        "-i", src_path,
        "-ss", f"{exact_seek}",
        "-t", f"{duration_sec}",
        "-c:v", encoder,
    ]
    if codec in _HW_H264_CODECS:
        # Premiere/Media Encoder "Match Source — Adaptive High Bitrate":
        # hardware H.264, VBR 1-pass, target scaled to the source pixel rate
        # (15.2 Mbps at 1080p23.976), High profile, Rec.709 tags, yuv420p.
        target = _adaptive_bitrate(src_path, resolution)
        rate_args = [
            "-b:v", str(target),
            "-maxrate", str(int(target * 1.5)),
            "-bufsize", str(int(target * 2)),
            "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            # Hardware encoders don't reliably copy the colour flags into the
            # H.264 VUI, so stamp them at the bitstream level too (AME writes
            # full Rec.709 signalling; players otherwise guess).
            "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0",
        ]
        if encoder == "h264_nvenc":
            if gpu_frames:
                # the frames are CUDA surfaces: no CPU pixel format
                i = rate_args.index("-pix_fmt")
                rate_args = rate_args[:i] + rate_args[i + 2:]
            cmd.extend(["-preset", "p5", "-profile:v", "high", "-rc", "vbr", *rate_args])
        elif encoder == "h264_videotoolbox":
            # VideoToolbox has no -preset/-rc flags: VBR is the default rate
            # mode once -b:v is set. -allow_sw keeps the rare machine without
            # the hardware encode block working (software VT path).
            cmd.extend(["-profile:v", "high", "-allow_sw", "1", *rate_args])
        else:
            cmd.extend(["-preset", "medium", "-profile:v", "high", *rate_args])
    elif codec in ("libx264", "libx265", "libsvtav1", "libvpx-vp9"):
        cmd.extend(["-crf", str(crf), "-preset", "medium"])
    elif codec == "prores_ks":
        # ProRes profile 3 = ProRes 422 HQ, the sane default for delivery.
        cmd.extend(["-profile:v", "3"])
    elif codec == "dnxhr":
        # DNxHR is a parametric codec — ffmpeg refuses to encode without an
        # explicit profile. `dnxhr_sq` is the 4:2:2 8-bit standard variant,
        # the most universal choice for editorial workflows.
        cmd.extend(["-profile:v", "dnxhr_sq", "-pix_fmt", "yuv422p"])

    height = {"1080p": 1080, "720p": 720, "480p": 480}.get(resolution) if resolution != "source" else None
    if gpu_frames:
        cmd.extend(["-vf", f"scale_cuda=-2:{height}:format=nv12" if height else "scale_cuda=format=nv12"])
    elif height:
        cmd.extend(["-vf", f"scale=-2:{height}"])

    if audio == "copy":
        cmd.extend(["-c:a", "copy"])
    elif audio == "encode":
        # AAC stereo at 48 kHz — matches the AME preset (320 kbps by default).
        kbps = max(96, min(512, int(audio_bitrate_kbps or 320)))
        cmd.extend(["-c:a", "aac", "-b:a", f"{kbps}k", "-ar", "48000", "-ac", "2"])
    elif audio == "mute":
        cmd.extend(["-an"])

    cmd.extend(["-map", "0:v:0", "-map", "0:a?", "-avoid_negative_ts", "make_zero", "-y", output_path])

    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=_creation_flags())
    if proc.returncode != 0 and gpu_frames:
        # A stream NVDEC can't take (old codec, odd profile): same export
        # with a software decode.
        log.warning("NVDEC export failed for %s (%s); retrying with a software decode",
                    src_path, (proc.stderr or "").strip()[-200:])
        return export_scene(src_path, start_ms, end_ms, output_path, codec=codec, crf=crf, audio=audio,
                            resolution=resolution, audio_bitrate_kbps=audio_bitrate_kbps,
                            _gpu_decode=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr}")
    return output_path


def _cuda_decode_available() -> bool:
    from ..indexing.cuts import _ffmpeg_hw_profile
    prof = _ffmpeg_hw_profile()
    return prof is not None and prof.get("name") == "cuda"


def generate_proxy(src_path: str, start_ms: int, end_ms: int, output_path: str,
                   *, quality: str = "medium") -> str:
    """Generate a small H.264 .mp4 (or VP9 .webm fallback) proxy for hover preview.

    Optimised for the use case: a silent, low-fps, low-res clip that loads
    instantly when the user mouses over a scene. We don't need audio (browser
    plays it muted on hover anyway), we don't need 30 fps (humans don't notice
    on a 320×180 thumb), and we don't need full vertical resolution.

    With h264_nvenc or h264_videotoolbox: ~30-50× realtime → ~5-10 s of
    encode for the full ~300 proxies of a 22-min episode. With the libx264
    fallback: ~10-15 s per proxy = several minutes per episode.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bitrate = _PROXY_BITRATES.get(quality, _PROXY_BITRATES["medium"])
    start_sec = start_ms / 1000.0
    duration_sec = max(0.05, (end_ms - start_ms) / 1000.0)

    hw_encoder = _hw_h264_encoder() if output_path.lower().endswith(".mp4") else None
    # On macOS the decoder follows the indexing rule (cuts._decode_profile):
    # ffmpeg's VideoToolbox decode is latency-bound and slower than the
    # threaded software decoder except for HEVC above 1440p (M1 Pro: 0.72 s
    # -> 0.39 s per 3 s proxy of a 1080p H.264 episode).
    decode_args = (_videotoolbox_decode_args(src_path)
                   if hw_encoder == "h264_videotoolbox" else [])
    scale_vf = "scale=-2:240,fps=15"
    if hw_encoder == "h264_nvenc":
        from ..indexing.cuts import _ffmpeg_hw_profile
        prof = _ffmpeg_hw_profile()
        if prof is not None and prof.get("name") == "cuda":
            # NVDEC + scale_cuda + NVENC: frames never leave the GPU. The
            # software 1080p decode was most of a per-scene proxy (~3x).
            decode_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            scale_vf = "scale_cuda=-2:240,fps=15"

    base = [
        _ffmpeg_path(),
        "-hide_banner", "-loglevel", "error",
        *decode_args,
        "-ss", f"{max(0, start_sec - 0.5)}",
        "-i", src_path,
        "-ss", f"{min(start_sec, 0.5)}",
        "-t", f"{duration_sec}",
    ]

    if hw_encoder == "h264_nvenc":
        # `p1` = fastest preset, `tune ll` = low-latency, ~30-50× realtime on
        # consumer cards. Muted, 240p, 15 fps is way enough for hover previews.
        cmd = base + [
            "-c:v", "h264_nvenc",
            "-preset", "p1",
            "-tune", "ll",
            "-b:v", bitrate,
            "-vf", scale_vf,
            "-an",
            "-y", output_path,
        ]
    elif hw_encoder == "h264_videotoolbox":
        # `-realtime 1` biases the encoder for speed over quality — the right
        # trade-off for a 240p hover preview.
        cmd = base + [
            "-c:v", "h264_videotoolbox",
            "-realtime", "1",
            "-allow_sw", "1",
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


# ── roto frame extraction (shared with the mask session) ────────────────────
def extract_png_frames(
    src_path: str,
    start_ms: int,
    end_ms: int,
    out_pattern: Path,
    *,
    fps: float,
    max_dim: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> List[Path]:
    """Decode a clip window to a PNG sequence on a deterministic frame timeline.

    This is THE canonical frame sampler for the roto pipeline: the mask
    session (``models.sam2.extract_clip_frames``) and the alpha export both
    go through it with identical seek + fps arguments, so mask index N and
    export frame N always come from the same source frame. Do not fork this
    command elsewhere — a second extraction with different seek/fps args is
    exactly how mattes drift out of sync with their footage.

    ``fps`` must be the source's probed average rate (``probe_video``): the
    fps filter is then a 1:1 pass-through on CFR sources and resamples VFR
    sources onto a stable CFR timeline that both consumers share.
    """
    filters = [f"fps={fps}"]
    if max_dim is not None:
        # Resize so the long side is at most `max_dim`. Lanczos preserves
        # anime line work better than the default bicubic scaler.
        filters.append(
            f"scale='if(gt(iw,ih),min(iw,{max_dim}),-2)':"
            f"'if(gt(iw,ih),-2,min(ih,{max_dim}))':flags=lanczos"
        )

    start_sec = max(0.0, start_ms / 1000.0)
    duration_sec = max(0.05, (end_ms - start_ms) / 1000.0)
    cmd = [
        _ffmpeg_path(),
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec}",
        "-i", src_path,
        "-t", f"{duration_sec}",
        "-vf", ",".join(filters),
    ]
    if max_frames is not None:
        cmd.extend(["-frames:v", str(int(max_frames))])
    cmd.extend(["-start_number", "0", str(out_pattern)])

    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=_creation_flags())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {proc.stderr[-800:]}")
    return sorted(out_pattern.parent.glob(f"*{out_pattern.suffix}"))


def _export_alpha_raw(src_path: str, start_ms: int, end_ms: int, masks: np.ndarray, output_path: str,
                      *, codec: str, fps: float, finish_frame, check_cancel=None) -> str:
    """``export_scene_with_alpha`` without the PNG round-trip (CUDA path).

    Same frames as :func:`extract_png_frames` — identical seek / ``-t`` /
    ``fps`` arguments, rgb24 out of the same swscale conversion — but piped
    raw, and the RGBA result is piped raw into the encoder: PNG compression of
    every 1080p RGBA frame was ~0.9 s/frame of the cutout job. The per-frame
    matte finishing runs on a few threads (OpenCV releases the GIL), in order.
    """
    from concurrent.futures import ThreadPoolExecutor

    from ..finish.video_io import probe_clip

    clip = probe_clip(src_path)
    w, h = clip.width, clip.height
    n = int(masks.shape[0])
    start_sec = max(0.0, start_ms / 1000.0)
    duration_sec = max(0.05, (end_ms - start_ms) / 1000.0)
    dec = subprocess.Popen(
        [_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
         "-ss", f"{start_sec}", "-i", src_path, "-t", f"{duration_sec}",
         "-vf", f"fps={fps}", "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_creation_flags(),
    )
    enc = subprocess.Popen(
        [_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-framerate", f"{fps}", "-i", "-",
         "-an", *_encoder_args_for_alpha(codec), "-y", output_path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_creation_flags(),
    )
    import threading
    err: list[bytes] = []
    drain = threading.Thread(target=lambda: err.extend(enc.stderr), daemon=True)
    drain.start()

    def frames():
        fb = w * h * 3
        for _ in range(n):
            buf = dec.stdout.read(fb)
            if len(buf) < fb:
                return
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()

    written = 0
    try:
        workers = max(1, min(4, (os.cpu_count() or 2) - 1))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="alpha-export") as pool:
            pending = []
            for i, rgb in enumerate(frames()):
                if check_cancel is not None:
                    check_cancel()
                pending.append(pool.submit(finish_frame, rgb, masks[i]))
                if len(pending) > workers * 2:
                    enc.stdin.write(pending.pop(0).result().tobytes())
                    written += 1
            for fut in pending:
                enc.stdin.write(fut.result().tobytes())
                written += 1
    except BrokenPipeError:
        pass
    finally:
        for f in (dec.stdout, enc.stdin):
            try:
                f.close()
            except Exception:
                pass
        if dec.poll() is None:
            dec.kill()
        dec.wait()
        enc.wait()
        drain.join(timeout=2)
    if written == 0:
        raise RuntimeError("ffmpeg produced no RGB frames for alpha export")
    if written != n and abs(written - n) > 1:
        log.warning("alpha export: %d source frames vs %d masks — encoded %d", written, n, written)
    if enc.returncode != 0:
        raise RuntimeError(f"ffmpeg alpha export failed: {b''.join(err).decode(errors='replace')[-800:]}")
    return output_path


def _encoder_args_for_alpha(codec: str) -> list[str]:
    if codec == "prores_4444_alpha":
        # Profile 4 = ProRes 4444 (10-bit + alpha). yuva444p10le is the pixel
        # format that actually carries the alpha channel — without it ffmpeg
        # would silently drop the alpha plane.
        return ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]
    if codec == "vp9_alpha":
        # libvpx-vp9 + yuva420p = VP9 with embedded alpha in WebM. The
        # `auto-alt-ref` flag must be off when emitting alpha, otherwise
        # libvpx silently strips it.
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


def alpha_partial_path(output_path: str) -> Path:
    """``<stem>.partial<suffix>``: where :func:`export_scene_with_alpha`
    encodes before renaming onto ``output_path`` (the suffix still picks the
    muxer)."""
    p = Path(output_path)
    return p.with_name(f"{p.stem}.partial{p.suffix}")


def export_scene_with_alpha(src_path: str, start_ms: int, end_ms: int, masks: np.ndarray,
                            output_path: str, *, check_cancel=None, **kw) -> str:
    """Atomic :func:`_encode_scene_with_alpha`: encodes to
    :func:`alpha_partial_path`, renamed onto ``output_path`` on success and
    deleted on failure or cancellation (``check_cancel`` is called per frame
    and may raise) — a failed export never leaves a truncated but playable
    file under the final name."""
    part = alpha_partial_path(output_path)
    part.parent.mkdir(parents=True, exist_ok=True)
    try:
        _encode_scene_with_alpha(src_path, start_ms, end_ms, masks, str(part),
                                 check_cancel=check_cancel, **kw)
        os.replace(part, output_path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return output_path


def _encode_scene_with_alpha(
    src_path: str,
    start_ms: int,
    end_ms: int,
    masks: np.ndarray,
    output_path: str,
    *,
    codec: str = "prores_4444_alpha",
    fps: float = 24.0,
    decontaminate_rgb: bool = False,
    edge_refine: bool = True,
    bg_aware_cleanup: bool = True,
    raw_pipes: bool = False,
    check_cancel=None,
) -> str:
    """Encode the clip with the roto masks as the alpha channel.

    ``codec`` is either ``prores_4444_alpha`` (ProRes 4444, 10-bit, ``.mov`` —
    Resolve/AE/Premiere all consume it natively) or ``vp9_alpha`` (VP9 + alpha
    in ``.webm`` — lighter, web-friendly, less universally supported).

    Single unified path: the source is re-decoded at full resolution through
    :func:`extract_png_frames` with the SAME fps/seek arguments the mask
    session used, so RGB frame i and mask i are guaranteed to come from the
    same source frame. The alpha is then upscaled to source resolution — with
    a guided filter that snaps the matte edge back onto full-res line work
    when ``edge_refine`` is on, plain bilinear otherwise — optionally
    RGB-decontaminated, and the RGBA sequence is encoded in one ffmpeg pass.

    (The previous implementation fed the native-fps source and a fixed-24fps
    mask PNG sequence to ``alphamerge``, which pairs frames blindly — any
    source that wasn't exactly 24 fps drifted progressively out of sync with
    its matte. ``fps`` here must be the session's probed rate.)
    """
    from PIL import Image
    from ..models.mask_postprocess import (
        clean_edges_with_background,
        decontaminate_rgb_with_alpha,
        upsample_alpha_guided,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if masks.ndim != 3:
        raise ValueError(f"masks must be (T, H, W), got shape {masks.shape}")

    def finish_frame(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)
        if alpha.shape != rgb.shape[:2]:
            if edge_refine:
                alpha = upsample_alpha_guided(alpha, rgb)
            else:
                alpha = _resize_alpha(alpha, (rgb.shape[1], rgb.shape[0]))
        # BG-aware cleanup re-solves the soft band against the actual local
        # background (kills the "old background aura" around the matte); the
        # legacy decontaminate is the weaker fallback that only recolours edge
        # RGB without touching alpha.
        if bg_aware_cleanup:
            # Colour-only: re-solving alpha here eroded cel interiors next to
            # ink lines (Mask Mode benchmark, 0.4).
            rgb, alpha = clean_edges_with_background(rgb, alpha, resolve_alpha=False)
        elif decontaminate_rgb:
            rgb = decontaminate_rgb_with_alpha(rgb, alpha)
        return np.dstack([rgb, (alpha * 255.0).round().astype(np.uint8)])

    if raw_pipes:
        return _export_alpha_raw(src_path, start_ms, end_ms, masks, output_path, codec=codec,
                                 fps=fps, finish_frame=finish_frame, check_cancel=check_cancel)

    with tempfile.TemporaryDirectory(prefix="amv_alpha_") as tmp:
        tmp_path = Path(tmp)
        rgb_dir = tmp_path / "rgb"
        rgba_dir = tmp_path / "rgba"
        rgb_dir.mkdir()
        rgba_dir.mkdir()

        frame_paths = extract_png_frames(
            src_path, start_ms, end_ms, rgb_dir / "%05d.png",
            fps=fps, max_frames=int(masks.shape[0]),
        )
        if not frame_paths:
            raise RuntimeError("ffmpeg produced no RGB frames for alpha export")

        n = min(len(frame_paths), int(masks.shape[0]))
        if abs(len(frame_paths) - masks.shape[0]) > 1:
            log.warning(
                "alpha export: %d source frames vs %d masks — encoding the first %d",
                len(frame_paths), masks.shape[0], n,
            )

        for i in range(n):
            if check_cancel is not None:
                check_cancel()
            with Image.open(frame_paths[i]) as im:
                rgb = np.asarray(im.convert("RGB"))
            rgba = finish_frame(rgb, masks[i])
            Image.fromarray(rgba, "RGBA").save(rgba_dir / f"{i:05d}.png", optimize=False)

        cmd = [
            _ffmpeg_path(),
            "-hide_banner", "-loglevel", "error",
            "-framerate", f"{fps}",
            "-i", str(rgba_dir / "%05d.png"),
            "-an",  # alpha exports for compositing don't carry audio
            *_encoder_args_for_alpha(codec),
            "-y", output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=_creation_flags())
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg alpha export failed: {proc.stderr[-800:]}")
    return output_path
