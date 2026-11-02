"""NVIDIA single pass without ffmpeg's filter graph: NVDEC -> CUDA tensors.

``onepass`` runs ffmpeg (NVDEC, then ``scale_cuda`` x3, ``overlay_cuda`` x2,
a download and a pipe for every frame). On the 6-vCPU test machine that
costs ~8.5 CPU-seconds per 4 minutes of 1080p H.264 — ~2 ms of CPU per frame
for ~0.5 ms of actual decoding — and the CPU, not the GPU, paced a season.

Here PyNvVideoCodec (MIT, optional) decodes with NVDEC straight into device
memory and torch does the downsizing in batches of frames:

* the 160x90 gray plane that drives cut detection and Search+ motion is
  computed with *exactly* ffmpeg 5.1's ``scale_cuda`` bicubic arithmetic
  (default A = 0: a 2-tap cubic Hermite here, point-sampled texels
  normalised k/255 to float32, centre geometry ``(x + 0.5) * scale - 0.5``,
  truncation to 8 bits): bit-identical planes, hence the same cuts;
* the 512x288 frame (median frames) follows the same formula; float
  contraction in nvcc makes ~2 % of its pixels differ by 1 level;
* frames are numbered like ``-fps_mode cfr`` (duplicate/drop beyond 1.1
  frames of drift), so timestamps match;
* everything is packed in ``onepass``'s stacked NV12 layout and handed to
  its scanner (cuts, medians, motion samples unchanged);
* the proxies' 240p copy: 15 fps frames encoded by NVENC from the GPU.

H.264 8-bit and HEVC 8/10-bit (10-bit samples arrive in 16 bits like
ffmpeg's p010, and follow its p010 -> nv12 kernel: 16-bit arithmetic, then
>> 8); anything else returns None and the caller uses the ffmpeg pass.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import cuts, onepass

log = logging.getLogger(__name__)

_BATCH = 32
PROXY_FPS = 15
_NVC = None


def _module():
    global _NVC
    if _NVC is None:
        try:
            extra = os.environ.get("AMV_PYNVVIDEOCODEC_PATH")
            if extra and extra not in sys.path:
                sys.path.insert(0, extra)
            import PyNvVideoCodec as nvc  # noqa: N813
            _NVC = nvc
        except Exception as exc:
            log.debug("PyNvVideoCodec unavailable: %s", exc)
            _NVC = False
    return _NVC or None


def available() -> bool:
    return os.environ.get("AMV_NVC_PASS", "1") != "0" and _module() is not None


def _buffers() -> int:
    """Decode buffers of ``_BATCH`` frames: the decode loop fills one while
    the worker downsizes and scans the others (``AMV_NVC_BUFFERS``)."""
    v = os.environ.get("AMV_NVC_BUFFERS", "")
    return max(2, min(8, int(v))) if v.isdigit() else 2


def _torch_cuda_handles(torch) -> tuple[int, int]:
    """(CUcontext, CUstream) torch uses on this thread, as integers."""
    import ctypes
    torch.zeros(1, device="cuda")            # make the primary context current
    lib = ctypes.CDLL("libcuda.so.1" if sys.platform != "win32" else "nvcuda.dll")
    ctx = ctypes.c_void_p()
    if lib.cuCtxGetCurrent(ctypes.byref(ctx)) != 0 or not ctx.value:
        raise RuntimeError("no current CUDA context")
    return int(ctx.value), int(torch.cuda.current_stream().cuda_stream)


def _raw_copier(stream, nbytes: int):
    """copy(dst_ptr, src_ptr): cuMemcpyDtoDAsync of one frame on ``stream``
    (the decoding thread has torch's context current), or None."""
    try:
        import ctypes
        lib = ctypes.CDLL("libcuda.so.1" if sys.platform != "win32" else "nvcuda.dll")
        fn = lib.cuMemcpyDtoDAsync_v2
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        handle = ctypes.c_void_p(int(stream.cuda_stream))
        size = ctypes.c_size_t(nbytes)

        def copy(dst: int, src: int) -> None:
            if fn(dst, src, size, handle) != 0:
                raise RuntimeError("cuMemcpyDtoDAsync failed")
        return copy
    except Exception as exc:
        log.debug("raw CUDA copies unavailable (%s); torch copies", exc)
        return None


def cfr_copies(t: float, fps: float, sync_opts: int) -> int:
    """How many output frames ``-fps_mode cfr`` makes of a frame at ``t``
    seconds when ``sync_opts`` frames are already out (ffmpeg 5.1
    do_video_out: drop below -1.1 frames of drift, duplicate above 1.1)."""
    delta = t * fps - sync_opts + 1.0
    if delta < -1.1:
        return 0
    if delta > 1.1:
        return int(round(delta))
    return 1


class _Scaler:
    """ffmpeg 5.1 ``scale_cuda`` (bicubic, default parameter) for one
    source/destination size, batched: (B, H, W[, C]) uint8 -> (B, h, w[, C])."""

    def __init__(self, torch, src_wh, dst_wh, device, bits: int = 8):
        self.torch = torch
        self.bits = bits
        W, H = src_wh
        w, h = dst_wh
        f32 = torch.float32
        hs = np.float32(np.float32(W) / np.float32(w))
        vs = np.float32(np.float32(H) / np.float32(h))
        xi = (np.arange(w, dtype=np.float32) + np.float32(0.5)) * hs - np.float32(0.5)
        yi = (np.arange(h, dtype=np.float32) + np.float32(0.5)) * vs - np.float32(0.5)
        px, py = np.floor(xi), np.floor(yi)
        fx, fy = (xi - px).astype(np.float32), (yi - py).astype(np.float32)

        def fma(a, b, c):   # float32 fused multiply-add, exact through float64
            return (np.float64(a) * np.float64(b) + np.float64(c)).astype(np.float32)

        def coeffs(x):      # bicubic_coeffs, A = 0, as nvcc compiles it
            one, two, three = np.float32(1.0), np.float32(2.0), np.float32(-3.0)
            y = fma((fma(two, x, three) * x).astype(np.float32), x, one)
            u = (one - x).astype(np.float32)
            z = fma((fma(two, u, three) * u).astype(np.float32), u, one)
            wv = (((one - np.float32(0.0)) - y).astype(np.float32) - z).astype(np.float32)
            return [np.zeros_like(x), y, z, wv]

        self.cx = [torch.from_numpy(c.astype(np.float32)).to(device) for c in coeffs(fx)]
        self.cy = [torch.from_numpy(c.astype(np.float32)).to(device) for c in coeffs(fy)]
        self.xs = [torch.from_numpy(np.clip(px + d, 0, W - 1).astype(np.int64)).to(device) for d in (-1, 0, 1, 2)]
        self.ys = [torch.from_numpy(np.clip(py + d, 0, H - 1).astype(np.int64)).to(device) for d in (-1, 0, 1, 2)]
        # texel -> float like the texture unit: k / 255 (k / 65535 for 16-bit
        # samples) correctly rounded
        top = 255.0 if bits == 8 else 65535.0
        self.lut = (torch.arange(int(top) + 1, dtype=torch.float64) / top).to(f32).to(device)
        self.factor = top
        # taps whose weight is 0 for every output (2-tap cases) are skipped
        self.taps_x = [i for i in range(4) if bool((self.cx[i] != 0).any())] or [1]
        self.taps_y = [i for i in range(4) if bool((self.cy[i] != 0).any())] or [1]
        self.xs_all = torch.cat([self.xs[i] for i in self.taps_x])
        self.ys_all = torch.cat([self.ys[j] for j in self.taps_y])
        self.cxs = [self.cx[i] for i in self.taps_x]
        self.cys = [self.cy[j] for j in self.taps_y]

    def __call__(self, p):
        """The kernel's arithmetic, bit for bit: texels normalised k/255 to
        float32, the first tap a float32 product, every further tap a fused
        multiply-add (nvcc contracts ``res += c * w``; ``torch.addcmul`` is
        one kernel compiled the same way: same single rounding), x taps then
        y taps, then ``* 255`` and truncation. Zero-weight taps add exactly
        nothing and are skipped. One gather per row tap bounds memory."""
        torch = self.torch
        chan = p.dim() == 4
        B = p.shape[0]
        nx = len(self.taps_x)
        h, w = self.cy[0].shape[0], self.cx[0].shape[0]
        cx = [c.view(1, 1, -1, 1) if chan else c.view(1, 1, -1) for c in self.cxs]
        res = None
        for jj, j in enumerate(self.taps_y):
            g = p.index_select(1, self.ys[j]).index_select(2, self.xs_all)
            idx = g.long() if self.bits == 8 else (g.to(torch.int32) & 0xFFFF).long()
            t = self.lut[idx]
            t = t.view(B, h, nx, w, -1) if chan else t.view(B, h, nx, w)
            r = t[:, :, 0] * cx[0]
            for k in range(1, nx):
                r = torch.addcmul(r, t[:, :, k], cx[k])
            cyj = self.cys[jj].view(1, -1, 1, 1) if chan else self.cys[jj].view(1, -1, 1)
            res = r * cyj if res is None else torch.addcmul(res, r, cyj)
        if self.bits == 8:
            return (res * 255.0).to(torch.uint8)
        # p010 -> nv12: truncated to 16 bits, then conv_10to8 (>> 8)
        return ((res * 65535.0).to(torch.int32) >> 8).to(torch.uint8)


def _bit_depth(path: str, reported: int) -> int:
    """The decoder's demuxer reports 8 bits for 10-bit HEVC in Matroska (the
    usual anime release): the frames then came as 16-bit samples read as
    8-bit ones and the pass failed. Trust the stream's pixel format."""
    try:
        with cuts.av.open(path) as c:
            pix = c.streams.video[0].codec_context.pix_fmt or ""
    except Exception:
        return reported
    if "p10" in pix or pix.endswith("10le") or pix.endswith("10be"):
        return 10
    if "p12" in pix or pix.endswith("12le"):
        return 12
    return reported


def detect_cuts_and_medians(path: str, threshold: float, frame_sink=None,
                            cancelled: Optional[Callable[[], bool]] = None):
    """Same contract as :func:`onepass.detect_cuts_and_medians`, or None when
    this pass does not handle the file (the caller then uses ffmpeg)."""
    nvc = _module()
    if nvc is None:
        return None
    import torch

    info = cuts.probe_video(path)
    duration_ms, fps = info["duration_ms"], info["fps"]
    if duration_ms <= 0 or not fps:
        return None
    try:
        dmx = nvc.CreateDemuxer(filename=path)
        depth = _bit_depth(path, dmx.BitDepth())
        codec = str(dmx.GetNvCodecId()).split(".")[-1].lower()
        if (codec, depth) not in (("h264", 8), ("hevc", 8), ("hevc", 10)):
            return None
        bits = 8 if depth == 8 else 16      # 10-bit comes as 16-bit samples (like p010)
        W, H = dmx.Width(), dmx.Height()
        # torch's own (primary) context and this thread's stream: separate
        # PyNvVideoCodec contexts next to torch work in other threads
        # crashed the process
        torch.cuda.init()
        ctx, _stream = _torch_cuda_handles(torch)
        # the decoder (and the copies out of its surfaces) get their own
        # stream: its synchronisations must not wait behind the downsizing
        # (high priority: the cut network's passes, on a low-priority stream,
        # must not hold up the decode -> downsize -> scan chain)
        dec_stream = torch.cuda.Stream(priority=-1)
        extra = {}
        if os.environ.get("AMV_NVC_LATENCY") == "low":     # measurements (display order kept)
            extra["latency"] = nvc.DisplayDecodeLatencyType.LOW
        dec = nvc.CreateDecoder(gpuid=0, codec=dmx.GetNvCodecId(), cudacontext=ctx,
                                cudastream=int(dec_stream.cuda_stream), usedevicememory=True, **extra)
    except Exception as exc:
        log.debug("PyNvVideoCodec cannot open %s: %s", path, exc)
        return None

    dev = torch.device("cuda")
    fw, fh, sh, gw, gh = onepass.FRAME_W, onepass.FRAME_H, onepass.STACK_H, onepass.GRAY_W, onepass.GRAY_H
    s_gray = _Scaler(torch, (W, H), (gw, gh), dev, bits)
    s_big = _Scaler(torch, (W, H), (fw, fh), dev, bits)
    s_uv = _Scaler(torch, (W // 2, H // 2), (fw // 2, fh // 2), dev, bits)
    pxw = (round(W * onepass.PROXY_SOURCE_H / H) // 2) * 2
    s_py = _Scaler(torch, (W, H), (pxw, onepass.PROXY_SOURCE_H), dev, bits)
    s_puv = _Scaler(torch, (W // 2, H // 2), (pxw // 2, onepass.PROXY_SOURCE_H // 2), dev, bits)
    sample = torch.uint8 if bits == 8 else torch.int16

    matrix, full_range = onepass._color_info(path)
    gray_lut = None if full_range else onepass._TV_TO_PC
    sc = onepass._Scanner(fps, threshold, frame_sink,
                          lambda b: onepass._nv12_to_rgb_gpu(b, matrix, full_range), cancelled, gray_lut,
                          (matrix, full_range))

    # the cut network's frames and colour histograms: made on the GPU for
    # the whole batch (no per-frame work on the CPU, the frames stay there)
    net_rgb = None
    if sc.color_info is not None:
        net_rgb = onepass.SmallRgbGpu(torch, dev, matrix, full_range, gray_lut)
        sc.color_info = None

    proxy = _ProxyWriter(path, pxw, onepass.PROXY_SOURCE_H, fps)
    # decode buffers: the decode loop fills one while a worker thread
    # downsizes, downloads and scans the others
    nbuf = _buffers()
    buffers = [torch.empty((_BATCH, H * 3 // 2, W), dtype=sample, device=dev) for _ in range(nbuf)]
    free = [threading.Semaphore(1) for _ in range(nbuf)]
    import queue as _queue
    work: _queue.Queue = _queue.Queue(maxsize=nbuf)
    stacked = torch.zeros((_BATCH, sh * 3 // 2, fw), dtype=torch.uint8, device=dev)
    staging = torch.empty((_BATCH, onepass.FRAME_BYTES), dtype=torch.uint8).pin_memory()
    staging_np = staging.numpy()
    errors: list = []
    out_idx = 0            # CFR frame index (ffmpeg -fps_mode cfr numbering)
    next_px = 0            # next 15 fps proxy slot

    def flush(batch, n: int):
        nonlocal next_px
        if n == 0:
            return
        y = batch[:n, :H]
        uv = batch[:n, H:].view(n, H // 2, W // 2, 2)
        st = stacked[:n]
        st[:, :fh] = s_big(y)
        st[:, fh:sh, :gw] = s_gray(y)
        st[:, sh:sh + fh // 2] = s_uv(uv).reshape(n, fh // 2, fw)
        net = net_rgb(st) if net_rgb is not None else None
        # wait for the GPU on a blocking event (the thread sleeps) instead of
        # the copy's own synchronisation, which spins a core while it waits
        ev = torch.cuda.Event(blocking=True)
        ev.record(torch.cuda.current_stream())
        ev.synchronize()
        staging[:n].copy_(st.reshape(n, -1))
        if net is not None:
            sc.detector.push_net(net[0], net[1].cpu().numpy())
        host = staging_np[:n]
        done = 0
        while done < n:
            view = sc.write_view(n - done)
            k = len(view)
            view[:] = host[done:done + k]
            sc.commit(k)
            done += k
        # proxies: the CFR frames falling on 15 fps slots
        first = sc.n - n
        picks = []
        while True:
            src = int(round(next_px * fps / PROXY_FPS))
            if src >= sc.n:
                break
            if src >= first:
                picks.append(src - first)
            next_px += 1
        if picks and proxy.ok:
            sel = batch[picks]
            py = s_py(sel[:, :H])
            puv = s_puv(sel[:, H:].view(len(picks), H // 2, W // 2, 2)).reshape(len(picks), -1, pxw)
            proxy.write(torch.cat([py, puv], 1), [first + p for p in picks])
        proxy.release(sc)

    work_stream = torch.cuda.Stream(priority=-1)
    finished: dict = {}

    def worker():
        while True:
            item = work.get()
            if item is None:
                with torch.cuda.stream(work_stream):
                    if not errors:
                        try:
                            # the last decisions, then the held proxy frames
                            # (encoded on this thread, like the others)
                            finished["scan"] = sc.finish(duration_ms)
                            proxy.release(sc, final=True)
                        except BaseException as exc:
                            errors.append(exc)
                    proxy.finish_encode()
                return
            k, n, ready = item
            try:
                if not errors:
                    with torch.cuda.stream(work_stream):
                        work_stream.wait_event(ready)
                        flush(buffers[k], n)
            except BaseException as exc:   # surfaced by the decode loop
                errors.append(exc)
            finally:
                free[k].release()

    wt = threading.Thread(target=worker, daemon=True, name="nvc-scan")
    wt.start()
    cur, filled = 0, 0

    def hand_over():
        nonlocal cur, filled
        ready = torch.cuda.Event()
        ready.record(dec_stream)
        work.put((cur, filled, ready))
        cur, filled = (cur + 1) % nbuf, 0
        free[cur].acquire()
        if errors:
            raise errors[0]

    frame_bytes = W * (H * 3 // 2) * (1 if bits == 8 else 2)
    copy = _raw_copier(dec_stream, frame_bytes)
    try:
        free[cur].acquire()
        sync_opts = 0
        tb = dmx.GetTimebaseNum() / float(dmx.GetTimebaseDen())
        t0 = None
        for pkt in dmx:
            for fr in dec.Decode(pkt):
                t0 = fr.timestamp if t0 is None else t0
                nb = cfr_copies((fr.timestamp - t0) * tb, fps, sync_opts)
                if nb == 0:
                    continue
                raw_ok = copy is not None and tuple(fr.shape) == (H * 3 // 2, W) \
                    and tuple(fr.strides) in ((W, 1), (W * 2, 2))
                for _ in range(nb):
                    if raw_ok:
                        # one async device copy (no per-frame torch machinery)
                        copy(buffers[cur].data_ptr() + filled * frame_bytes, fr.GetPtrToPlane(0))
                    else:
                        with torch.cuda.stream(dec_stream):
                            buffers[cur][filled].copy_(torch.from_dlpack(fr).view(sample))
                    filled += 1
                    if filled == _BATCH:
                        hand_over()
                sync_opts += nb
                out_idx += nb
        if filled:
            hand_over()
        work.put(None)
        wt.join()
        if errors:
            raise errors[0]
    except BaseException:
        errors.append(RuntimeError("stopped"))
        try:
            work.put_nowait(None)
        except Exception:
            pass
        proxy.close(ok=False)
        raise
    cuts_ms, medians, n_frames = finished["scan"] if "scan" in finished else sc.finish(duration_ms)
    proxy.close(ok=n_frames > 0)
    if n_frames == 0:
        raise RuntimeError("PyNvVideoCodec decoded no frame")
    scenes = [(cuts_ms[i], cuts_ms[i + 1]) for i in range(len(cuts_ms) - 1)
              if cuts_ms[i + 1] - cuts_ms[i] >= cuts._MIN_SCENE_MS]
    return scenes, medians


class _CAI:
    def __init__(self, shape, strides, ptr):
        self.__cuda_array_interface__ = {"shape": shape, "strides": strides, "data": (ptr, False),
                                         "typestr": "|u1", "version": 3}


class _GpuNV12:
    """A contiguous (h*3/2, w) NV12 CUDA tensor as PyNvVideoCodec's encoder
    input (Y plane, interleaved UV plane)."""

    def __init__(self, t, w: int, h: int):
        p = t.data_ptr()
        self._t = t
        self._planes = [_CAI((h, w, 1), (w, 1, 1), p), _CAI((h // 2, w // 2, 2), (w, 2, 1), p + w * h)]

    def cuda(self):
        return self._planes


class _ProxyWriter:
    """The proxies' 240p/15 fps frames, encoded by NVENC straight from the
    GPU frames (PyNvVideoCodec).

    Copy mode: a 240p copy (constant QP 20, GOP 30, like the ffmpeg pass's
    copy) that the proxy pass decodes and re-encodes at every shot.

    Direct mode (``onepass._proxy_direct``): the hover proxies themselves —
    the proxies' bitrate, an IDR frame on the first 15 fps frame of every
    shot, no other key frame — so the proxy pass only cuts the stream
    (``-c copy``). A frame is encoded once the cut detector can no longer
    put a cut before it (``margin`` frames later, ~60 MB of 240p frames
    held on the GPU); the IDR positions go to a JSON file next to the
    stream, and the proxy pass checks every shot edge against them
    (``onepass.direct_split_ok``)."""

    def __init__(self, video_path: str, w: int, h: int, fps: float = 0.0):
        self.ok = False
        self.enc = self.mux = None
        self.direct = None
        self.pending: list = []
        d = onepass._proxy_cache_dir
        if d is None:
            return
        Path(d).mkdir(parents=True, exist_ok=True)
        base = Path(d) / onepass._proxy_source_name(video_path)
        self.direct = dict(onepass._proxy_direct) if onepass._proxy_direct and fps > 0 else None
        if self.direct:
            self.dst = base.with_suffix(".direct.h264")
            self.meta_path = base.with_suffix(".direct.json")
            self.raw = base.with_suffix(".part.direct.h264")
            self.meta_path.unlink(missing_ok=True)
        else:
            self.dst = base
            self.raw = base.with_suffix(".part.h264")
        self.tmp = base.with_suffix(".part.mkv")
        self.dst.unlink(missing_ok=True)
        self.w, self.h, self.n, self.fps = w, h, 0, fps
        self.pending: list = []        # (source frame index, (n, h*3/2, w) tensor, row)
        self.idr: list[int] = []       # slots encoded as IDR frames (direct mode)
        self.prev_src = -1
        self.cut_frames: list[int] = []
        self._cuts_seen = 1            # cuts_ms[0] is the start, not a cut
        self.nvc = _module()
        self.ok = self.nvc is not None
        self.mux = None

    def _open(self) -> None:
        """Created on the thread that encodes, on torch's context and that
        thread's current stream (an encoder on its own context, fed from
        another thread, crashed)."""
        import torch
        ctx, stream = _torch_cuda_handles(torch)
        if self.direct:
            # what the proxy pass's ffmpeg asked NVENC for: P1, low latency,
            # VBR at the proxy quality's bitrate; key frames only where forced
            bps = int(self.direct["bitrate"])
            self.enc = self.nvc.CreateEncoder(self.w, self.h, "NV12", False, codec="h264", preset="P1",
                                              tuning_info="low_latency", rc="vbr", bitrate=str(bps),
                                              maxbitrate=str(2 * bps), gop="100000", idrperiod="100000",
                                              repeatspspps="1", fps=str(PROXY_FPS),
                                              cudacontext=ctx, cudastream=stream)
        else:
            self.enc = self.nvc.CreateEncoder(self.w, self.h, "NV12", False, codec="h264", preset="P1",
                                              rc="constqp", constqp="20", gop="30", fps=str(PROXY_FPS),
                                              cudacontext=ctx, cudastream=stream)
        self.mux = open(self.raw, "wb")

    def _mux(self, packets) -> None:
        for pk in packets:
            self.mux.write(bytes(pk["data"]) if isinstance(pk, dict) else bytes(pk))

    def _encode(self, frame, idr: bool = False) -> None:
        pp = self.nvc.NV_ENC_PIC_PARAMS()
        pp.inputTimeStamp = self.n
        if idr:
            pp.encodePicFlags = int(self.nvc.NV_ENC_PIC_FLAGS.FORCEIDR)
            self.idr.append(self.n)
        self.n += 1
        self._mux(self.enc.Encode(_GpuNV12(frame, self.w, self.h), pp))

    def write(self, frames, srcs=None) -> None:
        """frames: (n, h*3/2, w) uint8 CUDA tensor; srcs: the CFR source
        frame index of each (direct mode: held until :meth:`release`)."""
        try:
            if self.enc is None:
                self._open()
            frames = frames.contiguous()
            if self.direct:
                self.pending.extend((int(s), frames, i) for i, s in enumerate(srcs))
                return
            import torch
            torch.cuda.current_stream().synchronize()   # frames ready for NVENC
            for i in range(frames.shape[0]):
                self._encode(frames[i])
        except Exception as exc:
            # the proxy pass then decodes the 1080p source again: say so
            log.warning("240p proxy copy not written (NVENC: %s); the proxies will decode the source", exc)
            self.ok = False
            self.pending.clear()

    def release(self, sc, final: bool = False) -> None:
        """Direct mode: encode the held frames the detector is done with
        (all of them once the stream ended), IDR on each shot's first."""
        if not self.direct or not self.ok or not self.pending:
            return
        try:
            cuts_ms = sc.cuts_ms[:-1] if final else sc.cuts_ms   # finish() appends the duration
            for ms in cuts_ms[self._cuts_seen:]:
                self.cut_frames.append(sc.first_at(ms))
            self._cuts_seen = len(cuts_ms)
            limit = float("inf") if final else sc.n - (sc.detector.lookahead + 6 * _BATCH)
            k = 0
            while k < len(self.pending) and self.pending[k][0] < limit:
                k += 1
            if not k:
                return
            import bisect
            import torch
            torch.cuda.current_stream().synchronize()
            for src, frames, i in self.pending[:k]:
                # a shot starts on a frame in (previous slot's frame, this one]
                a = bisect.bisect_right(self.cut_frames, self.prev_src)
                idr = self.n == 0 or (a < len(self.cut_frames) and self.cut_frames[a] <= src)
                self._encode(frames[i], idr)
                self.prev_src = src
            del self.pending[:k]
        except Exception as exc:
            log.warning("hover proxies not encoded during the decode (NVENC: %s); "
                        "the proxy pass will decode the source", exc)
            self.ok = False
            self.pending.clear()

    def finish_encode(self) -> None:
        """EndEncode, on the encoding thread."""
        if self.enc is None or self.mux is None:
            return
        try:
            self._mux(self.enc.EndEncode())
            self.mux.close()
        except Exception:
            self.ok = False
        self.mux = None

    def close(self, ok: bool) -> None:
        if self.enc is None:
            return
        if self.mux is not None:
            try:
                self.mux.close()
            except Exception:
                pass
            self.ok = False
        if self.direct:
            if ok and self.ok and self.n and not self.pending and self.raw.is_file():
                import json
                self.raw.replace(self.dst)
                self.meta_path.write_text(json.dumps({
                    "fps": self.fps, "proxy_fps": PROXY_FPS, "slots": self.n, "idr": self.idr,
                    "bitrate": int(self.direct["bitrate"])}))
            self.raw.unlink(missing_ok=True)
            return
        if ok and self.ok and self.n and self.raw.is_file():
            # raw stream -> MKV at 15 fps (timestamps from 0), no re-encode
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            r = subprocess.run([cuts._ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-r", str(PROXY_FPS),
                                "-i", str(self.raw), "-c", "copy", "-f", "matroska", "-y", str(self.tmp)],
                               capture_output=True, creationflags=flags)
            if r.returncode == 0 and self.tmp.is_file():
                self.tmp.replace(self.dst)
            else:
                log.warning("240p proxy copy not muxed (ffmpeg %s: %s)", r.returncode,
                            (r.stderr or b"").decode("utf-8", "replace")[-300:])
        self.raw.unlink(missing_ok=True)
        self.tmp.unlink(missing_ok=True)
