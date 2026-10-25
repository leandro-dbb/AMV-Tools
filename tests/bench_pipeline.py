"""Full-pipeline benchmark: what the app does, end to end, on one machine.

Runs the real sidecar (``backend/server.py`` of ``--repo``, in its own
process with a fresh data folder) and drives it over HTTP exactly like the
UI does:

1. *season*: the episodes go into the index queue in one go, then
   ``/api/index/start`` — cuts, Search+ motion/intensity/palette, median
   frames, sub-segmentation, tags, embeddings, thumbnails, hover proxies
   (``generate_proxies`` on), and the Search+ background analysis if any
   scene was left without metrics;
2. *search*: the sidecar is restarted (a new session: the first query loads
   SigLIP 2 cold), then 10 text queries, 3 image queries and 5 "similar
   shots";
3. *export*: 10 kept clips (derush export with the app's export settings)
   and the same selection as FCP7 XML, FCPXML, EDL and OTIO timelines;
4. *finish*: on a 5 s clip, upscale x2 (animevideov3), RIFE x2 slow motion
   and depth (no matte: out of scope).

Measured: wall time of each part and of the whole pipeline, per-stage time
inside the indexing queue (from the sidecar's own progress events), mean
accelerator utilisation per stage (``nvidia-smi`` on NVIDIA; ``ioreg``'s
"Device Utilization %" on Apple Silicon, no sudo needed), mean CPU use, peak
VRAM / unified-memory GPU allocation, peak RAM of the sidecar process tree
(physical footprint on macOS) and swap activity.

    uv run python tests/bench_pipeline.py --season ep1.mkv ep2.mkv ep3.mkv \\
        --data-dir /tmp/amv-bench --out after.json --md after.md
    uv run python tests/bench_pipeline.py --quick extract-4min.mkv --data-dir /tmp/q --out q.json
    # an older checkout (the "before" column), same script and interpreter:
    uv run python tests/bench_pipeline.py --repo /path/to/old/checkout ...
    # tables (median of several runs per column) and output-quality checks:
    uv run python tests/bench_pipeline.py --compare before.json -- t1.json -- a1.json,a2.json
    uv run python tests/bench_pipeline.py --quality REF_DATA_DIR CUR_DATA_DIR

Models must already be in the Hugging Face cache (``HF_HOME``) and the
finishing models in ``AMV_MODELS_DIR``; missing finishing models skip that
job with a note.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent.parent

TEXT_QUERIES = (
    "goku", "beerus", "explosion", "close-up of a face", "gros plan visage",
    "two characters fighting in the sky", "night city", "someone eating", "laughing", "sunset",
)
SIMILAR_COUNT = 5
IMAGE_COUNT = 3
EXPORT_COUNT = 10
TIMELINE_FORMATS = ("fcp7xml", "fcpxml", "edl", "otio")


# ── system probes ───────────────────────────────────────────────────────────
class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64) for n in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


_libproc = None
if sys.platform == "darwin":
    try:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    except OSError:
        _libproc = None


def _footprint(pid: int) -> Optional[int]:
    """macOS physical footprint (includes Metal buffers that RSS misses)."""
    if _libproc is None:
        return None
    info = _RusageInfoV2()
    if _libproc.proc_pid_rusage(pid, 2, ctypes.byref(info)) == 0:
        return int(info.ri_phys_footprint)
    return None


def _processes() -> list[tuple[int, int, int]]:
    """(pid, ppid, rss bytes) of every process (no psutil in the app's venv)."""
    out = []
    if sys.platform.startswith("linux"):
        page = os.sysconf("SC_PAGE_SIZE")
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as fh:
                    st = fh.read().rsplit(")", 1)[1].split()
                out.append((int(d), int(st[1]), int(st[21]) * page))
            except (OSError, IndexError, ValueError):
                pass
        return out
    r = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=5)
    for line in r.stdout.splitlines():
        try:
            pid, ppid, rss = (int(x) for x in line.split()[:3])
            out.append((pid, ppid, rss * 1024))
        except ValueError:
            pass
    return out


def tree_memory(root_pid: int) -> int:
    """Bytes held by the sidecar and its children (ffmpeg, workers)."""
    procs = _processes()
    kids: dict[int, list[int]] = {}
    rss = {}
    for pid, ppid, r in procs:
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = r
    total, todo = 0, [root_pid]
    while todo:
        pid = todo.pop()
        if pid not in rss:
            continue
        fp = _footprint(pid)
        total += fp if fp is not None else rss[pid]
        todo.extend(kids.get(pid, []))
    return total


class _Cpu:
    """System-wide CPU use (%) since the previous call."""

    def __init__(self):
        self._prev = self._read()

    @staticmethod
    def _read():
        if sys.platform.startswith("linux"):
            with open("/proc/stat") as fh:
                v = [int(x) for x in fh.readline().split()[1:]]
            return sum(v), v[3] + v[4]
        return None

    def percent(self) -> Optional[float]:
        if sys.platform == "darwin":
            r = subprocess.run(["ps", "-A", "-o", "%cpu="], capture_output=True, text=True, timeout=5)
            tot = sum(float(x.replace(",", ".")) for x in r.stdout.split() if x)
            return min(100.0, tot / (os.cpu_count() or 1))
        cur = self._read()
        if cur is None or self._prev is None:
            return None
        dt, di = cur[0] - self._prev[0], cur[1] - self._prev[1]
        self._prev = cur
        return 100.0 * (1 - di / dt) if dt > 0 else None


def swap_counters() -> dict:
    """Cumulative swap-outs (pages) and swap in use (MB), system-wide."""
    out: dict[str, Any] = {}
    try:
        if sys.platform == "darwin":
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"Swapouts:\s+(\d+)", vm)
            out["swapouts"] = int(m.group(1)) if m else None
            su = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True,
                                timeout=5).stdout
            m = re.search(r"used = ([\d.,]+)M", su)
            out["swap_used_mb"] = float(m.group(1).replace(",", ".")) if m else None
        else:
            with open("/proc/vmstat") as fh:
                for line in fh:
                    if line.startswith("pswpout "):
                        out["swapouts"] = int(line.split()[1])
            with open("/proc/meminfo") as fh:
                mem = {ln.split(":")[0]: int(ln.split()[1]) for ln in fh if ":" in ln}
            out["swap_used_mb"] = round((mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / 1024, 1)
    except Exception:
        pass
    return out


class Sampler:
    """Samples accelerator utilisation, CPU, memory of the sidecar tree."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.samples: list[dict] = []   # {t, gpu, dec, enc, vram_mb, cpu, mem}
        self.kind = "none"
        self.root_pid: Optional[int] = None
        self._stop = threading.Event()
        self._gpu: dict = {}
        self._nvsmi: Optional[subprocess.Popen] = None
        if shutil.which("nvidia-smi"):
            self.kind = "nvidia-smi"
        elif sys.platform == "darwin" and shutil.which("ioreg"):
            self.kind = "ioreg"
        self._threads = [threading.Thread(target=self._run, daemon=True)]
        if self.kind == "nvidia-smi":
            self._threads.append(threading.Thread(target=self._nvidia, daemon=True))
        elif self.kind == "ioreg":
            self._threads.append(threading.Thread(target=self._ioreg, daemon=True))

    def start(self) -> "Sampler":
        self._cpu = _Cpu()
        for t in self._threads:
            t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._nvsmi is not None:
            self._nvsmi.terminate()

    def _nvidia(self) -> None:
        self._nvsmi = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.decoder,utilization.encoder,memory.used",
             "--format=csv,noheader,nounits", f"-lms={int(self.interval * 1000)}"],
            stdout=subprocess.PIPE, text=True)
        for line in self._nvsmi.stdout:  # type: ignore[union-attr]
            try:
                g, d, e, m = (float(x) for x in line.split(",")[:4])
                self._gpu = {"gpu": g, "dec": d, "enc": e, "vram_mb": m}
            except ValueError:
                continue
            if self._stop.is_set():
                return

    def _ioreg(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                                     capture_output=True, text=True, timeout=5).stdout
                u = re.search(r'"Device Utilization %"=(\d+)', out)
                m = re.search(r'"In use system memory"=(\d+)', out)
                self._gpu = {"gpu": float(u.group(1)) if u else None,
                             "vram_mb": round(int(m.group(1)) / 2**20, 1) if m else None}
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            s = {"t": time.time(), "cpu": self._cpu.percent(), **self._gpu}
            if self.root_pid:
                s["mem"] = tree_memory(self.root_pid)
            self.samples.append(s)

    def window(self, t0: float, t1: float) -> list[dict]:
        return [s for s in self.samples if t0 <= s["t"] <= t1]


def _mean(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(statistics.fmean(vals), 1) if vals else None


def _peak(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def stats_for(samples: list[dict]) -> dict:
    mem = _peak(s.get("mem") for s in samples)
    return {
        "gpu_util_mean": _mean(s.get("gpu") for s in samples),
        "nvdec_util_mean": _mean(s.get("dec") for s in samples),
        "nvenc_util_mean": _mean(s.get("enc") for s in samples),
        "cpu_mean": _mean(s.get("cpu") for s in samples),
        "gpu_mem_peak_mb": _peak(s.get("vram_mb") for s in samples),
        "ram_peak_mb": round(mem / 2**20) if mem else None,
        "samples": len(samples),
    }


# ── the sidecar ─────────────────────────────────────────────────────────────
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Sidecar:
    def __init__(self, repo: Path, python: str, data_dir: Path, log_path: Path):
        self.repo, self.python, self.data_dir, self.log_path = repo, python, data_dir, log_path
        self.port = _free_port()
        self.proc: Optional[subprocess.Popen] = None
        self.events: list[tuple[float, dict]] = []
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_stop = threading.Event()
        self._cv = threading.Condition()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 180.0) -> float:
        env = dict(os.environ, AMV_DATA_DIR=str(self.data_dir), PYTHONUNBUFFERED="1")
        env.pop("PYTHONPATH", None)
        t0 = time.time()
        self.proc = subprocess.Popen(
            [self.python, str(self.repo / "backend" / "server.py"), "--port", str(self.port)],
            cwd=str(self.repo), env=env, stdout=open(self.log_path, "ab"), stderr=subprocess.STDOUT,
            start_new_session=True)
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"sidecar exited with {self.proc.returncode}, see {self.log_path}")
            try:
                self.get("/api/status", timeout=2)
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("sidecar did not answer")
        ready = time.time() - t0
        self._ws_stop.clear()
        self._ws_thread = threading.Thread(target=self._ws, daemon=True)
        self._ws_thread.start()
        return ready

    def _ws(self) -> None:
        import asyncio

        import websockets

        async def run():
            async with websockets.connect(f"ws://127.0.0.1:{self.port}/ws/progress",
                                          max_size=None, ping_interval=None) as ws:
                while not self._ws_stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    with self._cv:
                        self.events.append((time.time(), json.loads(msg)))
                        self._cv.notify_all()

        try:
            asyncio.run(run())
        except Exception as exc:  # sidecar stopped
            if not self._ws_stop.is_set():
                print(f"[bench] progress socket closed: {exc}", flush=True)

    def wait_event(self, pred, since: int, timeout: float) -> Optional[dict]:
        deadline = time.time() + timeout
        with self._cv:
            while True:
                for _t, ev in self.events[since:]:
                    if pred(ev):
                        return ev
                left = deadline - time.time()
                if left <= 0:
                    return None
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(f"sidecar died (code {self.proc.returncode}), see {self.log_path}")
                self._cv.wait(min(left, 1.0))

    def stop(self) -> None:
        self._ws_stop.set()
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except Exception:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=5)
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=3)

    def _req(self, method: str, path: str, body: Any = None, timeout: float = 600) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            ctype = r.headers.get("Content-Type", "")
        return json.loads(raw) if "json" in ctype else raw

    def get(self, path: str, **kw) -> Any:
        return self._req("GET", path, **kw)

    def post(self, path: str, body: Any = None, **kw) -> Any:
        return self._req("POST", path, body if body is not None else {}, **kw)

    def put(self, path: str, body: Any, **kw) -> Any:
        return self._req("PUT", path, body, **kw)


# ── stages of the indexing queue, from the sidecar's progress messages ─────
_STAGES = (
    ("Probing", "probe"),
    ("Detecting cuts", "decode_cuts_motion"),
    ("Sub-segmenting", "sub_segmentation"),
    ("Extracting", "median_frames"),
    ("Inserting", "thumbnails_db"),
    ("Tagg", "tags"),
    ("Embedd", "embeddings"),
    ("Already indexed", "skip"),
    ("Done", "finalize"),
    ("Starting", "start"),
)


def _stage_of(ev: dict) -> Optional[str]:
    typ = ev.get("type")
    # index_queue's "idle" comes before the hover proxies (only "proxy"
    # events once per finished video follow).
    if typ in ("proxy", "idle"):
        return "hover_proxies"
    if typ == "analysis":
        return "searchplus_background"
    if typ != "indexing":
        return None
    msg = str(ev.get("message") or "")
    for prefix, name in _STAGES:
        if msg.startswith(prefix):
            return name
    return "other"


def stage_table(events: list[tuple[float, dict]], t_end: float, sampler: Sampler) -> dict:
    """Wall time per stage = time between a stage's event and the next
    stage's (sequential critical path as the app reports it)."""
    marks: list[tuple[float, str]] = []
    for t, ev in events:
        st = _stage_of(ev)
        if st and (not marks or marks[-1][1] != st):
            marks.append((t, st))
    spans: dict[str, list[tuple[float, float]]] = {}
    for i, (t, st) in enumerate(marks):
        t1 = marks[i + 1][0] if i + 1 < len(marks) else t_end
        spans.setdefault(st, []).append((t, t1))
    out = {}
    for st, sp in spans.items():
        samples = [s for a, b in sp for s in sampler.window(a, b)]
        out[st] = {"wall_s": round(sum(b - a for a, b in sp), 2), **stats_for(samples)}
    return out


# ── sections ────────────────────────────────────────────────────────────────
def _frames_of(path: str) -> Optional[int]:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "format=duration:stream=r_frame_rate", "-of", "json", path],
                             capture_output=True, text=True, timeout=30).stdout
        j = json.loads(out)
        num, den = j["streams"][0]["r_frame_rate"].split("/")
        return int(float(j["format"]["duration"]) * float(num) / float(den))
    except Exception:
        return None


def section_season(sc: Sidecar, sampler: Sampler, videos: list[str], res: dict) -> None:
    sc.put("/api/settings", {"indexing": {"generate_proxies": True}})
    t0 = time.time()
    since = len(sc.events)
    sc.post("/api/index/queue", {"paths": [{"path": v, "recursive": False} for v in videos]})
    sc.post("/api/index/start", {})
    # index_queue sends "idle", then the proxies run, then the worker's final
    # "idle": the second one ends the indexing run.
    first = sc.wait_event(lambda e: e.get("type") == "idle", since, 6 * 3600)
    if first is None:
        raise RuntimeError("indexing never finished")
    idx_first = next(i for i in range(since, len(sc.events)) if sc.events[i][1] is first)
    crashed = any(e.get("type") == "indexing" and "video" not in e
                  and str(e.get("message", "")).startswith("Failed") for _t, e in sc.events[since:])
    if not crashed:
        sc.wait_event(lambda e: e.get("type") == "idle", idx_first + 1, 3 * 3600)
    t_index = time.time()
    # Search+ metrics left over (normally none: computed during indexing).
    cov = sc.get("/api/searchplus/analysis").get("coverage", {})
    t_bg = t_index
    if cov.get("pending"):
        mark = len(sc.events)
        sc.post("/api/searchplus/analyze", {})
        while sc.get("/api/searchplus/analysis").get("running"):
            time.sleep(0.5)
        t_bg = time.time()
        sc.events.insert(mark, (t_index, {"type": "analysis"}))
    t1 = max(t_index, t_bg)
    window = sampler.window(t0, t1)
    vids = sc.get("/api/videos")
    vids = vids.get("videos", vids) if isinstance(vids, dict) else vids
    frames = sum(_frames_of(v) or 0 for v in videos)
    scenes = 0
    with sqlite3.connect(str(sc.data_dir / "amv_tools.db")) as conn:
        scenes = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
        proxies = conn.execute("SELECT COUNT(*) FROM scenes WHERE proxy_path IS NOT NULL").fetchone()[0]
        statuses = [r[0] for r in conn.execute("SELECT status FROM videos")]
    res["season"] = {
        "wall_s": round(t1 - t0, 2),
        "videos": len(videos), "statuses": statuses, "frames": frames, "scenes": scenes,
        "proxies": proxies, "metrics_pending_after_index": cov.get("pending"),
        "fps": round(frames / (t1 - t0), 1) if frames else None,
        **stats_for(window),
        "stages": stage_table([e for e in sc.events[since:]], t1, sampler),
        "events": [(round(t - t0, 2), e) for t, e in sc.events[since:]
                   if e.get("type") != "indexing" or e.get("percent") in (0, 1, 3, 100)
                   or not str(e.get("message", "")).startswith(("Tagged", "Embedded", "Sub-segmenting "))],
    }


def _scene_ids(sc: Sidecar) -> list[dict]:
    with sqlite3.connect(str(sc.data_dir / "amv_tools.db")) as conn:
        rows = conn.execute(
            "SELECT s.id, s.video_id, s.start_ms, s.end_ms FROM scenes s "
            "WHERE s.parent_scene_id IS NULL ORDER BY s.video_id, s.start_ms").fetchall()
    return [{"id": r[0], "video": r[1], "start": r[2], "end": r[3]} for r in rows]


def _spread(scenes: list[dict], n: int) -> list[dict]:
    """n scenes at fixed relative positions of each episode (independent of
    the models, so every build exports comparable clips)."""
    by_vid: dict[int, list[dict]] = {}
    for s in scenes:
        by_vid.setdefault(s["video"], []).append(s)
    vids = sorted(by_vid)
    out: list[dict] = []
    for k in range(n):
        lst = by_vid[vids[k % len(vids)]]
        dur = lst[-1]["end"]
        target = dur * (0.1 + 0.8 * ((k // len(vids)) + 0.5) / ((n + len(vids) - 1) // len(vids)))
        pick = min((s for s in lst if s not in out and 1500 <= s["end"] - s["start"] <= 8000),
                   key=lambda s: abs(s["start"] - target), default=None)
        if pick is None:
            pick = min((s for s in lst if s not in out), key=lambda s: abs(s["start"] - target))
        out.append(pick)
    return out


def section_search(sc: Sidecar, sampler: Sampler, res: dict, tmp: Path) -> None:
    out: dict[str, Any] = {}
    scenes = _scene_ids(sc)
    t0 = time.time()
    lat = []
    for i, q in enumerate(TEXT_QUERIES):
        a = time.time()
        r = sc.post("/api/search", {"query": q, "top_k": 200, "threshold": 0.05})
        lat.append(time.time() - a)
        if i == 0:
            out["first_text_cold_s"] = round(lat[0], 3)
            out["first_text_results"] = len(r.get("results", []))
    out["text_warm_ms_median"] = round(statistics.median(lat[1:]) * 1000, 1)
    # image queries: thumbnails of three scenes, as files (drag & drop)
    picks = _spread(scenes, IMAGE_COUNT + SIMILAR_COUNT)
    ilat = []
    for k, s in enumerate(picks[:IMAGE_COUNT]):
        img = tmp / f"query_{k}.jpg"
        img.write_bytes(sc.get(f"/api/scene/{s['id']}/thumbnail"))
        a = time.time()
        sc.post("/api/search", {"image_path": str(img), "top_k": 200, "threshold": 0.05})
        ilat.append(time.time() - a)
    out["image_ms"] = [round(x * 1000, 1) for x in ilat]
    slat = []
    for s in picks[IMAGE_COUNT:]:
        a = time.time()
        sc.get(f"/api/search/similar/{s['id']}")
        slat.append(time.time() - a)
    out["similar_ms"] = [round(x * 1000, 1) for x in slat]
    t1 = time.time()
    out["wall_s"] = round(t1 - t0, 2)
    out.update(stats_for(sampler.window(t0, t1)))
    res["search"] = out


def section_export(sc: Sidecar, sampler: Sampler, res: dict, tmp: Path) -> None:
    out: dict[str, Any] = {}
    picks = _spread(_scene_ids(sc), EXPORT_COUNT)
    for s in picks:
        sc.post("/api/derush/toggle", {"scene_id": s["id"]})
    out["clips_s"] = round(sum(s["end"] - s["start"] for s in picks) / 1000, 1)
    t0 = time.time()
    since = len(sc.events)
    sc.post("/api/derush/export", {"scope": "all"})
    ev = sc.wait_event(lambda e: e.get("type") == "derush_export" and e.get("done"), since, 3600)
    t1 = time.time()
    out["clips"] = {"wall_s": round(t1 - t0, 2), "ok": bool(ev and ev.get("ok")),
                    "message": ev.get("message") if ev else None, **stats_for(sampler.window(t0, t1))}
    tl: dict[str, Any] = {}
    a = time.time()
    for fmt in TIMELINE_FORMATS:
        b = time.time()
        r = sc.post("/api/timeline/export", {
            "format": fmt, "derush_scope": "all",
            "output_path": str(tmp / f"timeline_{fmt}"),
            "options": {"collect_media": False}})
        tl[fmt] = {"wall_s": round(time.time() - b, 3), "ok": bool(r.get("ok", True))}
    out["timelines"] = {"wall_s": round(time.time() - a, 3), **tl}
    out["wall_s"] = round(time.time() - t0, 2)
    out.update(stats_for(sampler.window(t0, time.time())))
    res["export"] = out


def section_finish(sc: Sidecar, sampler: Sampler, res: dict, source: str, start_ms: int,
                   tmp: Path, models: str = "") -> None:
    out: dict[str, Any] = {}
    src = {"path": source, "start_ms": start_ms, "end_ms": start_ms + 5000}
    up, interp, depth = ((models or ",,").split(",") + ["", "", ""])[:3]
    jobs = (
        ("upscale_x2", "upscale", {"model": up or "realesr-animevideov3", "scale": 2}),
        ("rife_x2", "interpolate", {"mode": "slowmo", "factor": 2, **({"model": interp} if interp else {})}),
        ("depth", "depth", {"format": "png16", **({"model": depth} if depth else {})}),
    )
    t_all = time.time()
    for name, kind, params in jobs:
        odir = tmp / f"finish_{name}"
        odir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            r = sc.post("/api/finish/jobs", {"kind": kind, "sources": [src],
                                             "params": {**params, "output_dir": str(odir)}})
        except urllib.error.HTTPError as exc:
            out[name] = {"skipped": f"HTTP {exc.code}: {exc.read()[:200]!r}"}
            continue
        jid = r["jobs"][0]["id"]
        job: dict = {}
        while True:
            job = next((j for j in sc.get("/api/finish/jobs")["jobs"] if j["id"] == jid), {})
            if job.get("status") in ("done", "completed", "failed", "error", "cancelled"):
                break
            time.sleep(0.25)
        t1 = time.time()
        out[name] = {"wall_s": round(t1 - t0, 2), "status": job.get("status"),
                     "error": job.get("error"), "output": job.get("output"),
                     **stats_for(sampler.window(t0, t1))}
        print(f"[bench] finish {name}: {out[name]['wall_s']} s ({job.get('status')})", flush=True)
    out["wall_s"] = round(time.time() - t_all, 2)
    res["finish"] = out


def environment(repo: Path) -> dict:
    env: dict[str, Any] = {"host": socket.gethostname(), "platform": platform.platform(),
                           "machine": platform.machine(), "cpus": os.cpu_count(), "python": sys.version.split()[0]}
    try:
        env["commit"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass
    env["commit"] = os.environ.get("AMV_BENCH_COMMIT") or env.get("commit")
    if shutil.which("nvidia-smi"):
        env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    elif sys.platform == "darwin":
        env["gpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True).stdout.strip()
    try:
        env["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return env


def run(args) -> dict:
    repo = Path(args.repo).resolve()
    data = Path(args.data_dir).resolve()
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    (data / ".setup_complete").write_text("bench\n")
    if args.settings:
        # Model/tier overrides, in place before the first launch (the sidecar
        # merges a partial settings file into its defaults).
        (data / "settings.json").write_text(args.settings)
    if args.autotune:
        # a measured autotune.json (same machine): the run uses its choices
        # instead of measuring in the background mid-bench
        shutil.copy(args.autotune, data / "autotune.json")
    tmp = data / "bench"
    tmp.mkdir()
    videos = [args.quick] if args.quick else list(args.season)
    res: dict[str, Any] = {"label": args.label, "mode": "quick" if args.quick else "full",
                           "videos": videos, "env": environment(repo), "started": time.strftime("%F %T")}
    sampler = Sampler().start()
    sw0 = swap_counters()
    sc = Sidecar(repo, args.python, data, data / "sidecar.log")
    t_all = time.time()
    try:
        res["sidecar_start_s"] = round(sc.start(), 2)
        sampler.root_pid = sc.proc.pid if sc.proc else None
        status = sc.get("/api/status")
        res["device"] = {k: status.get(k) for k in ("device", "gpu_name", "vram_gb", "tagger_provider")}
        print(f"[bench] sidecar up on {sc.port} ({res['device']})", flush=True)
        if "season" in args.sections:
            section_season(sc, sampler, videos, res)
            print(f"[bench] season: {res['season']['wall_s']} s, {res['season']['scenes']} scenes", flush=True)
        # A new session for the searches: SigLIP 2 loads on the first query.
        sc.stop()
        sc = Sidecar(repo, args.python, data, data / "sidecar.log")
        res["sidecar_restart_s"] = round(sc.start(), 2)
        sampler.root_pid = sc.proc.pid if sc.proc else None
        if "search" in args.sections:
            section_search(sc, sampler, res, tmp)
            print(f"[bench] search: {res['search']}", flush=True)
        if "export" in args.sections:
            section_export(sc, sampler, res, tmp)
            print(f"[bench] export: {res['export']['wall_s']} s", flush=True)
        if "finish" in args.sections and not args.quick:
            section_finish(sc, sampler, res, args.finish_source or videos[0], args.finish_start_ms, tmp,
                           args.finish_models)
    finally:
        sc.stop()
        sampler.stop()
    res["total_s"] = round(time.time() - t_all, 2)
    res["pipeline_s"] = round(sum(res.get(k, {}).get("wall_s", 0) for k in ("season", "search", "export", "finish")), 2)
    sw1 = swap_counters()
    res["swap"] = {"before": sw0, "after": sw1,
                   "swapouts_delta": (sw1.get("swapouts") - sw0.get("swapouts"))
                   if sw0.get("swapouts") is not None and sw1.get("swapouts") is not None else None}
    res["overall"] = stats_for(sampler.samples)
    res["sampler"] = sampler.kind
    return res


# ── reporting ───────────────────────────────────────────────────────────────
def _fmt_s(v) -> str:
    if v is None:
        return "–"
    v = float(v)
    if v >= 120:
        return f"{int(v // 60)} min {int(round(v % 60)):02d}"
    return f"{v:.1f} s" if v >= 10 else f"{v:.2f} s"


def _pct(v) -> str:
    return "–" if v is None else f"{v:.0f} %"


STAGE_ORDER = ("probe", "decode_cuts_motion", "sub_segmentation", "median_frames", "thumbnails_db",
               "tags", "embeddings", "finalize", "hover_proxies", "searchplus_background")


def markdown(res: dict) -> str:
    L = [f"## {res.get('label') or ''} — {res['env'].get('host')} ({res['env'].get('gpu', '')})", "",
         f"Pipeline total: **{_fmt_s(res.get('pipeline_s'))}** (season + search + export + finish); "
         f"run wall time incl. sidecar starts {_fmt_s(res.get('total_s'))}.", "",
         "| part | wall | GPU util | CPU | GPU mem peak | RAM peak |", "|---|---:|---:|---:|---:|---:|"]
    for k in ("season", "search", "export", "finish"):
        p = res.get(k)
        if not p:
            continue
        L.append(f"| {k} | {_fmt_s(p.get('wall_s'))} | {_pct(p.get('gpu_util_mean'))} | {_pct(p.get('cpu_mean'))} "
                 f"| {p.get('gpu_mem_peak_mb') or '–'} MB | {p.get('ram_peak_mb') or '–'} MB |")
    st = (res.get("season") or {}).get("stages", {})
    if st:
        L += ["", "| indexing stage | wall | GPU util | NVDEC | NVENC | CPU |", "|---|---:|---:|---:|---:|---:|"]
        for k in STAGE_ORDER + tuple(x for x in st if x not in STAGE_ORDER):
            if k in st:
                v = st[k]
                L.append(f"| {k} | {_fmt_s(v['wall_s'])} | {_pct(v.get('gpu_util_mean'))} | "
                         f"{_pct(v.get('nvdec_util_mean'))} | {_pct(v.get('nvenc_util_mean'))} | {_pct(v.get('cpu_mean'))} |")
    f = res.get("finish") or {}
    for k in ("upscale_x2", "rife_x2", "depth"):
        if k in f:
            L.append(f"- finish {k}: {_fmt_s(f[k].get('wall_s'))} ({f[k].get('status') or f[k].get('skipped')}), "
                     f"GPU {_pct(f[k].get('gpu_util_mean'))}")
    s = res.get("search") or {}
    if s:
        L.append(f"- search: first text query (cold) {s.get('first_text_cold_s')} s, warm text {s.get('text_warm_ms_median')} ms, "
                 f"image {s.get('image_ms')} ms, similar {s.get('similar_ms')} ms")
    L.append(f"- swap-outs during the run: {res.get('swap', {}).get('swapouts_delta')}")
    return "\n".join(L) + "\n"


def _median_res(runs: list[dict]) -> dict:
    """Median of numeric leaves across runs (same structure)."""
    def merge(vals):
        if all(isinstance(v, dict) for v in vals):
            keys = set().union(*vals)
            return {k: merge([v[k] for v in vals if k in v]) for k in keys}
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if nums and len(nums) == len(vals):
            return statistics.median(nums)
        return vals[0]
    return merge(runs) if len(runs) > 1 else runs[0]


def compare_markdown(groups: list[list[str]]) -> str:
    cols = [_median_res([json.loads(Path(p).read_text()) for p in g]) for g in groups]
    names = [c.get("label") or f"col{i}" for i, c in enumerate(cols)]
    L = ["| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(cols)]

    def row(label, get, fmt=_fmt_s, gain=True):
        vals = [get(c) for c in cols]
        cells = [fmt(v) for v in vals]
        if gain and vals[0] and vals[-1]:
            cells[-1] += f" (×{float(vals[0]) / float(vals[-1]):.1f})"
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    row("**pipeline total**", lambda c: c.get("pipeline_s"))
    for k in ("season", "search", "export", "finish"):
        row(k, lambda c, k=k: (c.get(k) or {}).get("wall_s"))
    for st in STAGE_ORDER:
        row(f"— {st}", lambda c, st=st: (((c.get("season") or {}).get("stages") or {}).get(st) or {}).get("wall_s"))
    for k in ("upscale_x2", "rife_x2", "depth"):
        row(f"finish {k}", lambda c, k=k: ((c.get("finish") or {}).get(k) or {}).get("wall_s"))
    row("first search (cold)", lambda c: (c.get("search") or {}).get("first_text_cold_s"))
    row("GPU util, season", lambda c: (c.get("season") or {}).get("gpu_util_mean"), _pct, False)
    for st in STAGE_ORDER:
        row(f"GPU util — {st}", lambda c, st=st: (((c.get("season") or {}).get("stages") or {}).get(st) or {}).get("gpu_util_mean"), _pct, False)
    row("RAM peak (MB)", lambda c: (c.get("overall") or {}).get("ram_peak_mb"), lambda v: "–" if v is None else f"{v:.0f}", False)
    row("GPU mem peak (MB)", lambda c: (c.get("overall") or {}).get("gpu_mem_peak_mb"), lambda v: "–" if v is None else f"{v:.0f}", False)
    row("swap-outs (pages)", lambda c: (c.get("swap") or {}).get("swapouts_delta"), lambda v: "–" if v is None else f"{v:.0f}", False)
    return "\n".join(L) + "\n"


# ── output quality between two runs (data folders) ──────────────────────────
def _db_dump(data_dir: Path) -> dict:
    import numpy as np
    conn = sqlite3.connect(str(Path(data_dir) / "amv_tools.db"))
    vids = {r[0]: Path(r[1]).name for r in conn.execute("SELECT id, filepath FROM videos")}
    out: dict[str, dict] = {v: {"cuts": [], "scenes": {}} for v in vids.values()}
    rows = conn.execute("SELECT id, video_id, start_ms, end_ms, parent_scene_id, embedding FROM scenes").fetchall()
    tags: dict[int, list[str]] = {}
    for sid, tag, _c in conn.execute("SELECT scene_id, tag, confidence FROM scene_tags ORDER BY confidence DESC"):
        tags.setdefault(sid, []).append(tag)
    try:
        motion = dict(conn.execute("SELECT scene_id, motion_type FROM scene_metrics").fetchall())
    except sqlite3.Error:
        motion = {}
    children = {r[4] for r in rows if r[4] is not None}
    for sid, vid, s, e, parent, emb in rows:
        v = out[vids[vid]]
        if parent is None:
            v["cuts"].append(s)
        v["scenes"][(s, e)] = {
            "emb": np.frombuffer(emb, dtype=np.float32) if emb else None,
            "tags": tags.get(sid, []), "motion": motion.get(sid), "leaf": sid not in children}
    for v in out.values():
        v["cuts"].sort()
    return out


def quality(ref_dir: str, cur_dir: str, fps: float = 24000 / 1001) -> dict:
    import numpy as np
    ref, cur = _db_dump(Path(ref_dir)), _db_dump(Path(cur_dir))
    res: dict[str, Any] = {}
    frame_ms = 1000.0 / fps
    for name, r in ref.items():
        c = cur.get(name)
        if c is None:
            res[name] = {"missing": True}
            continue
        rc, cc = r["cuts"], c["cuts"]
        exact = len(set(rc) & set(cc))
        # same frame: within half a frame
        within1 = sum(1 for t in rc if any(abs(t - u) < frame_ms * 0.6 for u in cc))
        common = [k for k in r["scenes"] if k in c["scenes"]]
        cos, top, motion_same, motion_n = [], [], 0, 0
        for k in common:
            a, b = r["scenes"][k], c["scenes"][k]
            if a["emb"] is not None and b["emb"] is not None and a["emb"].shape == b["emb"].shape:
                cos.append(float(np.dot(a["emb"], b["emb"]) /
                                 (np.linalg.norm(a["emb"]) * np.linalg.norm(b["emb"]) + 1e-12)))
            ta, tb = a["tags"][:5], b["tags"][:5]
            if ta or tb:
                top.append(len(set(ta) & set(tb)) / max(len(ta), len(tb)))
            if a["motion"] is not None and b["motion"] is not None:
                motion_n += 1
                motion_same += a["motion"] == b["motion"]
        res[name] = {
            "cuts_ref": len(rc), "cuts_cur": len(cc), "cuts_exact": exact, "cuts_same_frame": within1,
            "scenes_ref": len(r["scenes"]), "scenes_cur": len(c["scenes"]), "scenes_common": len(common),
            "emb_cos_min": round(min(cos), 5) if cos else None,
            "emb_cos_median": round(float(np.median(cos)), 5) if cos else None,
            "emb_below_0995": sum(1 for x in cos if x < 0.995),
            "tags_top5_agreement": round(float(np.mean(top)), 4) if top else None,
            "motion_same_class": round(motion_same / motion_n, 4) if motion_n else None,
        }
    return res


def finish_psnr(ref_dir: str, cur_dir: str) -> dict:
    """PSNR of each finishing output against the reference run's."""
    out = {}
    for name in ("upscale_x2", "rife_x2", "depth"):
        a = sorted((Path(ref_dir) / "bench" / f"finish_{name}").rglob("*.*"))
        b = sorted((Path(cur_dir) / "bench" / f"finish_{name}").rglob("*.*"))
        a = [p for p in a if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".png")]
        b = [p for p in b if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".png")]
        if not a or not b:
            out[name] = None
            continue
        if a[0].suffix.lower() == ".png":
            import numpy as np
            from PIL import Image
            vals = []
            for pa, pb in list(zip(a, b))[:: max(1, len(a) // 10)]:
                x = np.asarray(Image.open(pa), dtype=np.float64)
                y = np.asarray(Image.open(pb), dtype=np.float64)
                peak = 65535.0 if x.max() > 255 else 255.0
                mse = float(np.mean((x - y) ** 2))
                vals.append(99.0 if mse == 0 else 10 * np.log10(peak * peak / mse))
            out[name] = round(min(vals), 2)
            continue
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(b[0]), "-i", str(a[0]),
                            "-lavfi", "psnr", "-f", "null", "-"], capture_output=True, text=True)
        m = re.search(r"average:([\d.inf]+)", r.stderr)
        out[name] = m.group(1) if m else None
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", nargs="+", default=[], help="episodes indexed as one queue")
    p.add_argument("--quick", help="a single short extract; no finishing")
    p.add_argument("--repo", default=str(_HERE), help="checkout whose sidecar is benchmarked")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--data-dir", default="/tmp/amv-bench-pipeline")
    p.add_argument("--label", default="")
    p.add_argument("--sections", default="season,search,export,finish")
    p.add_argument("--finish-source", help="video for the finishing clip (default: first episode)")
    p.add_argument("--finish-start-ms", type=int, default=90_000)
    p.add_argument("--settings", help="JSON merged into the sidecar settings first (models, tier)")
    p.add_argument("--autotune", help="autotune.json copied into the data folder first (backend.autotune)")
    p.add_argument("--finish-models", default="",
                   help="upscale,interpolate,depth model ids for the finish jobs (empty = defaults)")
    p.add_argument("--out")
    p.add_argument("--md")
    p.add_argument("--compare", nargs="+", help="JSON files; '--' separates columns, commas = runs of one column")
    p.add_argument("--quality", nargs=2, metavar=("REF_DATA_DIR", "CUR_DATA_DIR"))
    a = p.parse_args(argv)
    a.sections = set(a.sections.split(","))
    return a


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--compare" in argv:
        i = argv.index("--compare")
        groups, cur = [], []
        for x in argv[i + 1:]:
            if x == "--":
                groups.append(cur)
                cur = []
            else:
                cur.extend(y for y in x.split(",") if y)
        groups.append(cur)
        print(compare_markdown([g for g in groups if g]))
        return 0
    args = parse_args(argv)
    if args.quality:
        q = {"index": quality(*args.quality), "finish_psnr_db": finish_psnr(*args.quality)}
        print(json.dumps(q, indent=1, default=str))
        if args.out:
            Path(args.out).write_text(json.dumps(q, indent=1, default=str))
        return 0
    if not args.quick and not args.season:
        print("give --season EP... or --quick CLIP", file=sys.stderr)
        return 2
    res = run(args)
    md = markdown(res)
    print(md)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    if args.md:
        Path(args.md).write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
