"""One decode for cut detection *and* median frames (software decode path).

The regular pipeline decodes every video twice: once at 160×90 gray for the
cut detector (+ Search+ motion analysis), then again from the first median
to the end at 512×288 RGB to pull one median frame per scene. With hardware
decode + on-device scaling (NVDEC/scale_cuda) the second pass is cheap; with
the software decoder used on Apple Silicon (see ``cuts._decode_profile``) it
costs as much as the first one.

Here a single ffmpeg process emits, for every 4th frame only (the cut
detector's own stride), one rgb24 picture made of two stacked parts:

    rows   0..287  the 512×288 RGB frame (same scaler as ``frames.py``,
                   bicubic on the software path)
    rows 288..377  the 160×90 gray plane, produced by *exactly* the filter
                   chain of the two-pass cut detector, padded to 512 wide

so cut decisions and motion statistics are bit-identical to the two-pass
path, and the median frame of each scene is picked from the RGB part as the
scene closes. Only frames that can still become the median of the open
scene are buffered: the median of a scene that started at ``s`` and is still
running at ``now`` is at least ``s + (now - s) / 2``, so older samples are
dropped — memory is half the current scene, ~110 KB/s of footage.

Median precision: the sampled frame nearest to the exact median, i.e. within
half a stride (≤ 2 frames, ~83 ms at 23.976 fps). Measured against container
timestamps on a real episode, the two-pass extraction is not exact either
(+1 frame on a quarter of the scenes, from its seek-relative frame count).
"""
from __future__ import annotations

import logging
import subprocess
import sys
from collections import deque
from typing import Callable, Optional

import numpy as np
from PIL import Image

from . import cuts

log = logging.getLogger(__name__)

_RGB_W, _RGB_H = 512, 288
_MAX_SCENE_SAMPLES = 900
_W = max(_RGB_W, cuts._DIFF_W * cuts._FRAME_STRIDE)   # stacked picture width


def supported(path: str) -> bool:
    """True when the video decodes on the software profile (Apple Silicon),
    where a second full decode is the expensive part of indexing."""
    prof = cuts._decode_profile(path)
    return prof is not None and prof["name"] == "software"


def _command(path: str, fps: float, profile: dict, proxy_out=None) -> list[str]:
    gw, gh = cuts._DIFF_W, cuts._DIFF_H
    k = cuts._FRAME_STRIDE
    gray = profile["scale_vf"].format(w=gw, h=gh) + ",format=gray"
    rgb = profile.get("scale_vf_rgb", profile["scale_vf"]).format(w=_RGB_W, h=_RGB_H) + ",format=rgb24"
    proxy_graph, proxy_args = "", []
    if proxy_out is not None:
        # third branch: the hover-proxy source (see ``proxy_source``)
        from . import proxy_source
        proxy_graph = f";[p]{proxy_source.filter_chain()}[px]"
        proxy_args = ["-map", "[px]", *proxy_source.output_args(proxy_out)]
    graph = (
        f"[0:v]fps={fps:.6f},split={3 if proxy_out is not None else 2}[a][b]"
        f"{'[p]' if proxy_out is not None else ''};"
        # every frame's gray plane, grouped by stride: slot 0 is the sampled
        # frame the detector decides on, slots 1..3 locate the exact cut
        f"[a]{gray},tile=layout={k}x1,format=rgb24,pad={_W}:{gh}[g];"
        f"[b]select='not(mod(n\\,{k}))',{rgb},pad={_W}:{_RGB_H}[c];"
        f"[c][g]vstack=inputs=2[out]"
        f"{proxy_graph}"
    )
    return [
        cuts._ffmpeg_path(),
        "-hide_banner", "-nostats", "-loglevel", "error",
        "-fflags", "+discardcorrupt",
        *profile["input_args"],
        "-i", path,
        "-an", "-sn",
        "-filter_complex", graph, "-map", "[out]",
        # select drops 3 frames in 4: a CFR muxer would duplicate them back
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
        *proxy_args,
    ]


def detect_cuts_with_medians(
    path: str,
    threshold: float = 0.30,
    frame_sink: Optional[Callable[[int, np.ndarray], None]] = None,
    on_median: Optional[Callable[[int, Image.Image], None]] = None,
    on_scene: Optional[Callable[[int, int, list], None]] = None,
    proxy_source: bool = False,
) -> tuple[list[tuple[int, int]], dict[int, Image.Image]]:
    """Same contract as :func:`cuts.detect_cuts` plus the median frame of
    every returned scene: ``(scenes, {median_ms: PIL image})`` where
    ``median_ms = s + (e - s) // 2`` (the pipeline's ``_median_ms``).
    ``on_median(median_ms, image)`` is called as soon as each scene closes,
    while decoding continues.

    ``on_scene(start_ms, end_ms, samples)`` (sub-segmentation) receives every
    sample of the closed scene as ``[(ts_ms, rgb ndarray), ...]`` — the
    caller picks sub-segment frames from them, so no other decode is needed.
    Samples are then kept for the whole open scene instead of its second
    half; past ``_MAX_SCENE_SAMPLES`` (~2.5 min of footage, ~400 MB) every
    other one is dropped, which keeps a long take bounded.

    ``proxy_source``: the same decode also records the 240p hover-proxy
    source of the video (see ``proxy_source``), kept only if the pass
    completes."""
    info = cuts.probe_video(path)
    duration_ms, fps = info["duration_ms"], info["fps"] or 30.0
    if duration_ms <= 0:
        return [], {}
    profile = cuts._decode_profile(path)
    if profile is None:
        raise RuntimeError("single-pass decode needs an ffmpeg decode profile")

    gw, gh = cuts._DIFF_W, cuts._DIFF_H
    stride = cuts._FRAME_STRIDE
    frame_bytes = _W * (_RGB_H + gh) * 3
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proxy_out = None
    if proxy_source:
        from . import proxy_source as _ps
        try:
            proxy_out = _ps.temp_path(path)
        except Exception as exc:
            log.debug("no proxy source for %s: %s", path, exc)
    proc = subprocess.Popen(_command(path, fps, profile, proxy_out), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, creationflags=flags,
                            bufsize=frame_bytes * 4)

    cuts_ms: list[int] = [0]
    medians: dict[int, Image.Image] = {}
    buf: deque[tuple[int, np.ndarray]] = deque()   # (ts, rgb) still eligible
    prev_hist: Optional[np.ndarray] = None
    prev_group: Optional[np.ndarray] = None     # gray planes of the previous stride
    sample_idx = -1
    last_detect_ms = 0

    def exact_cut(group_before: np.ndarray, first_after: np.ndarray, sample: int) -> int:
        """The detector compares frames ``stride`` apart; the shot change is
        one of the frames in between. Pick the largest consecutive histogram
        jump among them so the scene starts on its first frame (otherwise
        the previous scene ends with up to stride-1 frames of the next
        shot — a flash at the end of every exported clip)."""
        frames_ = [group_before[:, i * gw:(i + 1) * gw] for i in range(stride)] + [first_after]
        hists = [cuts._hist_from_plane_bytes(np.ascontiguousarray(f).tobytes(), gw, gh) for f in frames_]
        jumps = [float(np.abs(hists[i + 1] - hists[i]).sum()) for i in range(stride)]
        first = (sample - 1) * stride + 1 + int(np.argmax(jumps))
        return int(first / fps * 1000)

    def close_scene(start: int, end: int) -> None:
        if end - start < cuts._MIN_SCENE_MS:
            return
        target = start + (end - start) // 2
        if buf:
            pick = min(buf, key=lambda item: abs(item[0] - target))[1]
            medians[target] = Image.fromarray(pick)
            if on_median is not None:
                on_median(target, medians[target])
            if on_scene is not None:
                on_scene(start, end, list(buf))

    reached_end = False
    try:
        assert proc.stdout is not None
        read = proc.stdout.read
        while True:
            raw = read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                reached_end = True
                break
            sample_idx += 1
            ts_ms = int((sample_idx * cuts._FRAME_STRIDE / fps) * 1000)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(_RGB_H + gh, _W, 3)
            group = arr[_RGB_H:, :gw * stride, 0]
            gray = np.ascontiguousarray(group[:, :gw])
            hist = cuts._hist_from_plane_bytes(gray.tobytes(), gw, gh)
            if frame_sink is not None:
                frame_sink(ts_ms, gray)
            if prev_hist is not None:
                diff = float(np.abs(hist - prev_hist).sum()) * 0.5
                # gap rule on the sampled instants (same decisions as the
                # two-pass detector); only the reported position is refined
                if diff > threshold and ts_ms > last_detect_ms + cuts._MIN_CUT_GAP_MS:
                    last_detect_ms = ts_ms
                    cut_ms = max(cuts_ms[-1] + 1, exact_cut(prev_group, gray, sample_idx))
                    close_scene(cuts_ms[-1], cut_ms)
                    cuts_ms.append(cut_ms)
                    buf.clear()
            prev_hist = hist
            prev_group = group.copy()
            buf.append((ts_ms, np.ascontiguousarray(arr[:_RGB_H, :_RGB_W])))
            # Drop samples that can no longer be the nearest to the median of
            # the open scene (that median only moves forward as it grows).
            if on_scene is not None:
                if len(buf) > _MAX_SCENE_SAMPLES:
                    buf = deque(list(buf)[::2])
                continue
            start = cuts_ms[-1]
            floor = start + (ts_ms - start) // 2
            while len(buf) > 2 and buf[1][0] <= floor:
                buf.popleft()
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        if reached_end and proxy_out is not None:
            # the proxy source is still being finalised (mp4 index)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

    if proxy_out is not None:
        from . import proxy_source as _ps
        if proc.returncode == 0:
            _ps.commit(path)
        else:
            _ps.discard(path)
    if proc.returncode not in (0, None):
        err = ""
        try:
            err = (proc.stderr.read() or b"").decode("utf-8", errors="replace")[-500:]
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {err!r}")
    if sample_idx < 0:
        raise RuntimeError("ffmpeg produced no frames")

    close_scene(cuts_ms[-1], duration_ms)
    cuts_ms.append(duration_ms)
    scenes = [(cuts_ms[i], cuts_ms[i + 1]) for i in range(len(cuts_ms) - 1)
              if cuts_ms[i + 1] - cuts_ms[i] >= cuts._MIN_SCENE_MS]
    return scenes, medians
