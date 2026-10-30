"""One shot being rotoscoped in the editor, saved on disk as it changes.

Layout of ``<user data>/roto_projects/<id>/``::

    project.json             shot, layers, locked frames, edge settings, exports
    frames/00000.png         working-size frames (what the models read)
    jpg/00000.jpg            the same as JPEG (the UI, the SAM 2 video predictor)
    layers/<layer>/ai/*.png  the AI pass (8-bit matte), kept for the eraser
    layers/<layer>/cur/*.png the current matte (AI pass + the user's corrections)

Mattes live in memory as ``(T, H, W)`` uint8 per layer; only the frames a
change touched are rewritten (``save``), from a background thread, so a
correction never waits on the disk.

Every change goes through :meth:`RotoProject.apply`, which records an undo
step (before / after of the touched frames, zlib-compressed) and the frames
it locks. A *locked* frame was corrected or validated by the user: the AI
pass and the propagation never write over it.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
from PIL import Image

from .edges import EdgeSettings

log = logging.getLogger(__name__)

LAYER_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#eab308", "#a855f7", "#f97316", "#06b6d4", "#ec4899"]
HISTORY_MAX = 60


def projects_root() -> Path:
    from ..paths import user_data_dir
    p = user_data_dir() / "roto_projects"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pack(a: np.ndarray) -> bytes:
    return zlib.compress(np.ascontiguousarray(a, np.uint8).tobytes(), 1)


def _unpack(b: bytes, shape) -> np.ndarray:
    return np.frombuffer(zlib.decompress(b), np.uint8).reshape(shape).copy()


def to_u8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    if a.dtype == bool:
        return a.astype(np.uint8) * 255
    return np.clip(np.rint(a.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)


@dataclass
class Step:
    """One undoable change: the touched frames of one layer, before / after,
    and the locked set before / after."""
    layer: str
    label: str
    frames: List[int]
    before: Dict[int, bytes]
    after: Dict[int, bytes]
    locked_before: Set[int]
    locked_after: Set[int]
    ai_before: Optional[Dict[int, bytes]] = None
    ai_after: Optional[Dict[int, bytes]] = None


@dataclass
class RotoProject:
    dir: Path
    meta: dict
    frame_paths: List[Path]
    jpg_paths: List[Path]
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    ai: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    locked: Dict[str, Set[int]] = field(default_factory=dict)
    settings: EdgeSettings = field(default_factory=EdgeSettings)
    version: int = 0
    frame_versions: Dict[int, int] = field(default_factory=dict)
    undo_stack: List[Step] = field(default_factory=list)
    redo_stack: List[Step] = field(default_factory=list)
    features: Dict[int, dict] = field(default_factory=dict)     # combined-matte suspect features per frame
    lock: threading.RLock = field(default_factory=threading.RLock)
    _dirty: Set[tuple] = field(default_factory=set)
    _save_timer: Optional[threading.Timer] = None
    _rgb_cache: Dict[int, np.ndarray] = field(default_factory=dict)

    # ── basics ──────────────────────────────────────────────────────────────
    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def n(self) -> int:
        return len(self.frame_paths)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.meta["width"]), int(self.meta["height"])

    @property
    def layers(self) -> List[dict]:
        return self.meta["layers"]

    def layer_ids(self) -> List[str]:
        return [l["id"] for l in self.layers]

    def check_layer(self, layer: Optional[str]) -> str:
        if not layer:
            return self.layers[0]["id"]
        if layer not in self.masks:
            raise KeyError(f"unknown layer {layer}")
        return layer

    def has_ai_pass(self) -> bool:
        return any(bool(v) for v in self.meta.get("ai_done", {}).values())

    def rgb(self, t: int) -> np.ndarray:
        """Working-size RGB frame (small cache: the frames around the cursor)."""
        t = int(t)
        c = self._rgb_cache.get(t)
        if c is None:
            with Image.open(self.frame_paths[t]) as im:
                c = np.asarray(im.convert("RGB"))
            if len(self._rgb_cache) > 24:
                self._rgb_cache.pop(next(iter(self._rgb_cache)))
            self._rgb_cache[t] = c
        return c

    def matte(self, layer: str, t: int) -> np.ndarray:
        return self.masks[layer][t].astype(np.float32) / 255.0

    def combined_u8(self, t: int) -> np.ndarray:
        out = None
        for lid in self.layer_ids():
            m = self.masks[lid][t]
            out = m.copy() if out is None else np.maximum(out, m)
        return out

    def locked_any(self) -> Set[int]:
        s: Set[int] = set()
        for v in self.locked.values():
            s |= v
        return s

    # ── changes ─────────────────────────────────────────────────────────────
    def apply(self, layer: str, new: Dict[int, np.ndarray], *, label: str,
              lock: Iterable[int] = (), unlock: Iterable[int] = (), ai: bool = False,
              record: bool = True) -> None:
        """Write the mattes ``new`` (frame -> matte) into ``layer``, lock /
        unlock frames, record one undo step. ``ai``: the AI pass itself (the
        ``ai`` copy is replaced too)."""
        with self.lock:
            cur = self.masks[layer]
            frames = sorted(int(t) for t in new)
            locked_before = set(self.locked[layer])
            before = {t: _pack(cur[t]) for t in frames} if record else {}
            ai_before = None
            if ai and record:
                prev_ai = self.ai.get(layer)
                ai_before = {t: _pack(prev_ai[t]) if prev_ai is not None else b"" for t in frames}
            for t in frames:
                cur[t] = to_u8(new[t])
            if ai:
                if self.ai.get(layer) is None:
                    self.ai[layer] = np.zeros_like(cur)
                for t in frames:
                    self.ai[layer][t] = cur[t]
                self.meta.setdefault("ai_done", {})[layer] = True
            self.locked[layer] |= {int(t) for t in lock}
            self.locked[layer] -= {int(t) for t in unlock}
            if record:
                self.undo_stack.append(Step(
                    layer, label, frames, before, {t: _pack(cur[t]) for t in frames},
                    locked_before, set(self.locked[layer]), ai_before,
                    {t: _pack(cur[t]) for t in frames} if ai else None))
                del self.undo_stack[:-HISTORY_MAX]
                self.redo_stack.clear()
            self._touched(layer, frames)

    def set_locked(self, layer: str, t: int, locked: bool) -> None:
        with self.lock:
            before = set(self.locked[layer])
            if locked:
                self.locked[layer].add(int(t))
            else:
                self.locked[layer].discard(int(t))
            self.undo_stack.append(Step(layer, "lock", [], {}, {}, before, set(self.locked[layer])))
            self.redo_stack.clear()
            self.schedule_save()

    def _touched(self, layer: str, frames: Iterable[int]) -> None:
        self.version += 1
        for t in frames:
            self.frame_versions[int(t)] = self.version
            self.features.pop(int(t), None)
            self.features.pop(int(t) + 1, None)   # object_lost compares with the previous frame
            self._dirty.add((layer, int(t)))
        self.schedule_save()

    def _swap(self, step: Step, undo: bool) -> int:
        cur = self.masks[step.layer]
        src = step.before if undo else step.after
        h, w = cur.shape[1:]
        for t in step.frames:
            cur[t] = _unpack(src[t], (h, w))
        ai_src = step.ai_before if undo else step.ai_after
        if ai_src is not None:
            if self.ai.get(step.layer) is None:
                self.ai[step.layer] = np.zeros_like(cur)
            for t in step.frames:
                b = ai_src.get(t, b"")
                self.ai[step.layer][t] = _unpack(b, (h, w)) if b else 0
                self._dirty.add(("ai:" + step.layer, t))
        self.locked[step.layer] = set(step.locked_before if undo else step.locked_after)
        self._touched(step.layer, step.frames)
        return step.frames[0] if step.frames else -1

    def undo(self) -> Optional[int]:
        with self.lock:
            if not self.undo_stack:
                return None
            step = self.undo_stack.pop()
            f = self._swap(step, True)
            self.redo_stack.append(step)
            return f

    def redo(self) -> Optional[int]:
        with self.lock:
            if not self.redo_stack:
                return None
            step = self.redo_stack.pop()
            f = self._swap(step, False)
            self.undo_stack.append(step)
            return f

    # ── layers ──────────────────────────────────────────────────────────────
    def add_layer(self, name: Optional[str] = None) -> str:
        with self.lock:
            lid = uuid.uuid4().hex[:8]
            i = len(self.layers)
            self.layers.append({"id": lid, "name": name or f"Personnage {i + 1}",
                                "color": LAYER_COLORS[i % len(LAYER_COLORS)]})
            w, h = self.size
            self.masks[lid] = np.zeros((self.n, h, w), np.uint8)
            self.ai[lid] = None
            self.locked[lid] = set()
            self.version += 1
            self.features.clear()
            self.schedule_save()
            return lid

    def remove_layer(self, lid: str) -> None:
        with self.lock:
            if len(self.layers) <= 1:
                raise ValueError("a project keeps at least one layer")
            self.meta["layers"] = [l for l in self.layers if l["id"] != lid]
            self.masks.pop(lid, None)
            self.ai.pop(lid, None)
            self.locked.pop(lid, None)
            self.meta.get("ai_done", {}).pop(lid, None)
            self.undo_stack = [s for s in self.undo_stack if s.layer != lid]
            self.redo_stack = [s for s in self.redo_stack if s.layer != lid]
            self.version += 1
            self.features.clear()
            shutil.rmtree(self.dir / "layers" / lid, ignore_errors=True)
            self.schedule_save()

    # ── suspects ────────────────────────────────────────────────────────────
    def suspects(self) -> List[dict]:
        from .suspects import decide, frame_features
        with self.lock:
            if not self.has_ai_pass():
                return []
            prev = None
            feats = []
            for t in range(self.n):
                cur = self.combined_u8(t)
                f = self.features.get(t)
                if f is None:
                    f = frame_features(cur, prev)
                    self.features[t] = f
                feats.append(f)
                prev = cur
            return decide(feats, self.locked_any(), self.meta.get("shots"))

    # ── persistence ─────────────────────────────────────────────────────────
    def schedule_save(self, delay: float = 0.4) -> None:
        if self._save_timer is not None:
            self._save_timer.cancel()
        t = threading.Timer(delay, self._save_quiet)
        t.daemon = True
        self._save_timer = t
        t.start()

    def _save_quiet(self) -> None:
        try:
            self.save()
        except Exception:
            log.exception("roto project %s: autosave failed", self.id)

    def save(self) -> None:
        with self.lock:
            dirty, self._dirty = self._dirty, set()
            self.meta["locked"] = {k: sorted(v) for k, v in self.locked.items()}
            self.meta["settings"] = self.settings.to_dict()
            self.meta["updated"] = time.time()
            meta = json.dumps(self.meta, indent=1)
            snaps = []
            for lid, t in dirty:
                if lid.startswith("ai:"):
                    real = lid[3:]
                    arr = self.ai.get(real)
                    if arr is not None:
                        snaps.append((self.dir / "layers" / real / "ai" / f"{t:05d}.png", arr[t].copy()))
                    continue
                if lid in self.masks:
                    snaps.append((self.dir / "layers" / lid / "cur" / f"{t:05d}.png", self.masks[lid][t].copy()))
                    if self.ai.get(lid) is not None:
                        snaps.append((self.dir / "layers" / lid / "ai" / f"{t:05d}.png", self.ai[lid][t].copy()))
        for path, arr in snaps:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr, "L").save(path, compress_level=1)
        tmp = self.dir / "project.json.tmp"
        tmp.write_text(meta)
        tmp.replace(self.dir / "project.json")

    # ── create / load ───────────────────────────────────────────────────────
    @classmethod
    def create(cls, frame_paths: List[Path], *, fps: float, name: str, scene_id: Optional[int] = None,
               video_path: Optional[str] = None, start_ms: int = 0, end_ms: int = 0,
               root: Optional[Path] = None, move_frames: bool = False) -> "RotoProject":
        """New project from extracted working-size frames (copied — or moved
        with ``move_frames`` — into the project folder)."""
        pid = uuid.uuid4().hex[:12]
        d = (root or projects_root()) / pid
        (d / "frames").mkdir(parents=True)
        (d / "jpg").mkdir()
        new_paths, jpgs = [], []
        for i, fp in enumerate(frame_paths):
            dst = d / "frames" / f"{i:05d}.png"
            if move_frames:
                shutil.move(str(fp), dst)
            else:
                shutil.copyfile(fp, dst)
            new_paths.append(dst)
            with Image.open(dst) as im:
                rgb = im.convert("RGB")
                if i == 0:
                    w, h = rgb.size
                j = d / "jpg" / f"{i:05d}.jpg"
                rgb.save(j, quality=92)
                jpgs.append(j)
        meta = {
            "id": pid, "scene_id": scene_id, "name": name, "video_path": video_path,
            "start_ms": int(start_ms), "end_ms": int(end_ms), "fps": float(fps),
            "width": int(w), "height": int(h), "n_frames": len(new_paths),
            "layers": [{"id": "main", "name": "Personnage 1", "color": LAYER_COLORS[0]}],
            "locked": {"main": []}, "ai_done": {}, "settings": EdgeSettings().to_dict(),
            "exports": None, "created": time.time(), "updated": time.time(),
        }
        p = cls(dir=d, meta=meta, frame_paths=new_paths, jpg_paths=jpgs)
        p.masks["main"] = np.zeros((len(new_paths), h, w), np.uint8)
        p.ai["main"] = None
        p.locked["main"] = set()
        p.save()
        return p

    @classmethod
    def load(cls, d: Path) -> "RotoProject":
        meta = json.loads((d / "project.json").read_text())
        n = int(meta["n_frames"])
        frames = [d / "frames" / f"{i:05d}.png" for i in range(n)]
        jpgs = [d / "jpg" / f"{i:05d}.jpg" for i in range(n)]
        p = cls(dir=d, meta=meta, frame_paths=frames, jpg_paths=jpgs,
                settings=EdgeSettings.from_dict(meta.get("settings")))
        w, h = p.size
        for l in meta["layers"]:
            lid = l["id"]
            arr = np.zeros((n, h, w), np.uint8)
            cur = d / "layers" / lid / "cur"
            if cur.exists():
                for f in cur.glob("*.png"):
                    i = int(f.stem)
                    if 0 <= i < n:
                        with Image.open(f) as im:
                            arr[i] = np.asarray(im.convert("L"))
            p.masks[lid] = arr
            ai_dir = d / "layers" / lid / "ai"
            if meta.get("ai_done", {}).get(lid) and ai_dir.exists():
                ai = np.zeros_like(arr)
                for f in ai_dir.glob("*.png"):
                    i = int(f.stem)
                    if 0 <= i < n:
                        with Image.open(f) as im:
                            ai[i] = np.asarray(im.convert("L"))
                p.ai[lid] = ai
            else:
                p.ai[lid] = None
            p.locked[lid] = set(int(x) for x in meta.get("locked", {}).get(lid, []))
        return p


# ── registry of open projects ──────────────────────────────────────────────
_open: Dict[str, RotoProject] = {}
_open_lock = threading.Lock()


def get_project(pid: str, root: Optional[Path] = None) -> Optional[RotoProject]:
    with _open_lock:
        p = _open.get(pid)
        if p is not None:
            return p
        d = (root or projects_root()) / pid
        if not (d / "project.json").exists() or "/" in pid or "\\" in pid or pid.startswith("."):
            return None
        p = RotoProject.load(d)
        _open[pid] = p
        return p


def register(p: RotoProject) -> None:
    with _open_lock:
        _open[p.id] = p


def find_projects(scene_id: int, root: Optional[Path] = None) -> List[dict]:
    """Saved projects of a scene (``project.json`` metadata)."""
    out = []
    base = root or projects_root()
    for f in base.glob("*/project.json"):
        try:
            meta = json.loads(f.read_text())
        except Exception:
            continue
        if meta.get("scene_id") == scene_id:
            out.append(meta)
    out.sort(key=lambda m: -float(m.get("updated", 0)))
    return out
