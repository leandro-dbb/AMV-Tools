"""Music → shots: plan beat-aligned slots, then fill them with indexed shots.

Pure functions over plain dicts (no SQLite, no models) so the planner can be
unit-tested with hand-made analyses and shot lists.  :mod:`.library` loads
the real shots and query scores from the database.

Slots
    Each section is cut into shots lasting a whole number of beats: long
    shots on calm passages, shots that get shorter through a build-up, 1-2
    beats on drops.  Cut points are beat times; on drops they move onto a
    strong attack (kick, impact) when one is within 70 ms.

Shots
    Each slot asks for an intensity (the music's smoothed energy there);
    shots are ranked by how close their own intensity (camera + character
    motion, flashes — the Search+ metrics) is to it, by how their motion
    suits the passage, by the text/character wishes, and never reuse the
    same moment of a shot when "avoid reuse" is on.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

# Beats per shot for each passage at medium density.
BASE_BEATS = {
    "intro": 8, "verse": 4, "build": 4, "drop": 2, "chorus": 2, "bridge": 8, "outro": 8,
}
CALM_TYPES = ("intro", "bridge", "outro")
LOUD_TYPES = ("drop", "chorus")
MIN_SHOT_S = 0.36     # faster than this reads as flicker, not as cuts
MAX_SHOT_S = 8.0
ONSET_SNAP_MS = 70
EDGE_MARGIN_MS = 120  # stay clear of the very first/last frames of a shot
ALTERNATIVES = 6


@dataclass
class Slot:
    slot: int
    section_index: int
    section_type: str
    start_ms: int
    end_ms: int
    beats: int
    target: float = 0.5

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class Constraints:
    groups: list[str] = field(default_factory=list)
    characters: list[int] = field(default_factory=list)
    density: float = 0.5
    avoid_reuse: bool = True
    query: str = ""
    section_queries: dict[int, str] = field(default_factory=dict)


# ── slots ───────────────────────────────────────────────────────────────────
def density_factor(density: float) -> float:
    """0 (calm) → shots twice as long, 0.5 → as is, 1 (nervous) → half."""
    d = min(1.0, max(0.0, float(density)))
    return 2.0 ** ((0.5 - d) * 2.0)


def _beats_per_shot(kind: str, period_s: float, density: float, progress: float = 0.0) -> int:
    n = BASE_BEATS.get(kind, 4) * density_factor(density)
    if kind == "build":
        # Accelerating: 4 → 2 → 1 beats over the build-up.
        n = n / (1.0 if progress < 0.5 else (2.0 if progress < 0.8 else 4.0))
    choices = (1, 2, 4, 8, 16)
    n = min(choices, key=lambda c: abs(math.log2(c) - math.log2(max(0.5, n))))
    while n > 1 and n * period_s > MAX_SHOT_S:
        n //= 2
    while n * period_s < MIN_SHOT_S and n < 16:
        n *= 2
    return int(n)


def _energy_at(analysis: dict, start_ms: int, end_ms: int) -> float:
    values = analysis.get("energy", {}).get("values") or []
    step = analysis.get("energy", {}).get("step_ms") or 100
    if not values:
        return 0.5
    a = max(0, min(len(values) - 1, start_ms // step))
    b = max(a + 1, min(len(values), int(math.ceil(end_ms / step))))
    return float(np.mean(values[a:b]))


def plan_slots(analysis: dict, density: float = 0.5) -> list[Slot]:
    """Cut the whole song into beat-aligned slots, section by section."""
    duration = int(analysis["duration_ms"])
    beats = np.asarray(analysis.get("beats_ms") or [], dtype=np.int64)
    strong = np.asarray(analysis.get("strong_onsets_ms") or [], dtype=np.int64)
    sections = analysis.get("sections") or [
        {"index": 0, "type": "verse", "start_ms": 0, "end_ms": duration}]
    bpm = float(analysis.get("bpm") or 120.0)
    period_s = 60.0 / max(30.0, bpm)
    slots: list[Slot] = []

    for sec in sections:
        s0, s1 = int(sec["start_ms"]), int(sec["end_ms"])
        if s1 <= s0:
            continue
        inner = beats[(beats > s0) & (beats < s1)]
        grid = [s0] + [int(b) for b in inner]
        span = max(1, s1 - s0)
        i = 0
        cuts = [s0]
        sizes = []
        while i < len(grid):
            progress = (grid[i] - s0) / span
            n = _beats_per_shot(sec["type"], period_s, density, progress)
            j = i + n
            if j >= len(grid):
                break
            cut = grid[j]
            if sec["type"] in LOUD_TYPES and strong.size:
                # Cut on the impact when one is right there.
                k = int(np.argmin(np.abs(strong - cut)))
                if abs(int(strong[k]) - cut) <= ONSET_SNAP_MS and cuts[-1] < strong[k] < s1:
                    cut = int(strong[k])
            cuts.append(cut)
            sizes.append(n)
            i = j
        # Leftover shorter than half a shot joins the previous one.
        remaining = len(grid) - i
        n_last = _beats_per_shot(sec["type"], period_s, density, 1.0)
        if len(cuts) > 1 and remaining < max(1, n_last // 2) and (s1 - cuts[-1]) < 0.5 * n_last * period_s * 1000:
            cuts.pop()
            sizes[-1] += remaining
        else:
            sizes.append(remaining)
        cuts.append(s1)
        for a, b, n in zip(cuts, cuts[1:], sizes):
            if b <= a:
                continue
            slots.append(Slot(slot=len(slots), section_index=int(sec["index"]),
                              section_type=sec["type"], start_ms=int(a), end_ms=int(b),
                              beats=max(1, int(n)), target=_energy_at(analysis, a, b)))
    # Close any gap at the song's edges.
    if slots:
        slots[0].start_ms = 0
        slots[-1].end_ms = duration
    return slots


# ── shots ───────────────────────────────────────────────────────────────────
def _rank01(values: list[Optional[float]]) -> list[float]:
    """Percentile rank (0-1) of each value; None → 0.5."""
    known = sorted(v for v in values if v is not None)
    if not known:
        return [0.5 for _ in values]
    out = []
    n = len(known)
    for v in values:
        if v is None:
            out.append(0.5)
            continue
        lo = np.searchsorted(known, v, side="left")
        hi = np.searchsorted(known, v, side="right")
        out.append(float((lo + hi) / 2) / max(1, n - 1) if n > 1 else 0.5)
    return [min(1.0, max(0.0, x)) for x in out]


def _motion_fit(kind: str, shot: dict) -> float:
    activity = float(shot.get("activity") or 0.0)
    magnitude = float(shot.get("motion_magnitude") or 0.0)
    mtype = shot.get("motion_type") or "static"
    flashes = float(shot.get("flash_count") or 0)
    if kind in CALM_TYPES:
        return 1.0 - 0.6 * activity - 0.4 * min(1.0, magnitude) - (0.3 if mtype == "shake" else 0.0)
    if kind == "build":
        return 0.5 * min(1.0, magnitude) + 0.3 * activity + (0.2 if mtype in ("zoom_in", "pan_left", "pan_right", "tilt_up") else 0.0)
    if kind in LOUD_TYPES:
        return (0.5 * activity + 0.2 * min(1.0, flashes / 2.0)
                + (0.3 if mtype in ("shake", "zoom_in") else 0.0))
    return 0.5 - 0.3 * abs(activity - 0.4)


@dataclass
class _Usage:
    windows: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    def overlaps(self, sid: int, a: int, b: int) -> bool:
        return any(a < y and x < b for x, y in self.windows.get(sid, []))

    def count(self, sid: int) -> int:
        return len(self.windows.get(sid, []))

    def add(self, sid: int, a: int, b: int) -> None:
        self.windows.setdefault(sid, []).append((a, b))


def _window(shot: dict, length: int, usage: _Usage, avoid_reuse: bool) -> Optional[tuple[int, int]]:
    """A [in, out) range of `length` ms inside the shot, avoiding used ones."""
    s, e = int(shot["start_ms"]), int(shot["end_ms"])
    room = (e - s) - length
    if room < 0:
        return None
    margin = min(EDGE_MARGIN_MS, room // 2)
    lo, hi = s + margin, e - margin - length
    if hi < lo:
        lo = hi = s + room // 2
    # Middle first (the action is rarely on the very first frames), then
    # spread out positions so a long shot can serve several slots.
    candidates = [lo + (hi - lo) // 2, lo, hi]
    if hi - lo > length:
        steps = max(2, (hi - lo) // max(1, length))
        candidates += [lo + k * (hi - lo) // steps for k in range(1, steps)]
    for a in candidates:
        if not avoid_reuse or not usage.overlaps(int(shot["scene_id"]), a, a + length):
            return (int(a), int(a + length))
    return None


def _score(shot: dict, slot: Slot, target_rank: float, query_scores: dict, prev_sid: Optional[int],
           rng: random.Random) -> float:
    inten = shot["_rank"]
    score = 1.0 - abs(inten - target_rank)                      # 0..1
    score += 0.45 * _motion_fit(slot.section_type, shot)
    q = query_scores.get(slot.section_index)
    if q is None:
        q = query_scores.get("global")
    if q:
        score += 0.9 * float(q.get(int(shot["scene_id"]), 0.0))
    if shot.get("_character"):
        score += 0.35
    if prev_sid is not None and int(shot["scene_id"]) == prev_sid:
        score -= 1.0
    score += rng.uniform(0.0, 0.12)                             # variety on regenerate
    return score


def _clip(slot: Slot, shot: dict, window: tuple[int, int], score: float, locked: bool = False) -> dict:
    return {
        "slot": slot.slot,
        "section_index": slot.section_index,
        "start_ms": slot.start_ms,
        "end_ms": slot.end_ms,
        "beats": slot.beats,
        "scene_id": int(shot["scene_id"]),
        "video_id": int(shot["video_id"]),
        "video_name": shot.get("video_name") or "",
        "group": shot.get("group"),
        "src_in_ms": int(window[0]),
        "src_out_ms": int(window[1]),
        "intensity": None if shot.get("intensity") is None else round(float(shot["intensity"]), 3),
        # Where the shot sits in *this* library (0 calmest … 1 most intense):
        # what the matching uses, and what the UI should show next to target.
        "intensity_rank": round(float(shot.get("_rank", 0.5)), 3),
        "motion_type": shot.get("motion_type"),
        "target": round(float(slot.target), 3),
        "score": round(float(score), 3),
        "locked": locked,
        "alternatives": [],
    }


def _alt(shot: dict, window: tuple[int, int], score: float) -> dict:
    return {
        "scene_id": int(shot["scene_id"]),
        "video_id": int(shot["video_id"]),
        "video_name": shot.get("video_name") or "",
        "group": shot.get("group"),
        "src_in_ms": int(window[0]),
        "src_out_ms": int(window[1]),
        "intensity": None if shot.get("intensity") is None else round(float(shot["intensity"]), 3),
        "intensity_rank": round(float(shot.get("_rank", 0.5)), 3),
        "motion_type": shot.get("motion_type"),
        "score": round(float(score), 3),
    }


def _fit_locked(slots: list[Slot], locked: list[dict]) -> list[Slot]:
    """Make room for locked clips: drop the slots they cover, trim the ones
    they overlap, and keep everything contiguous."""
    if not locked:
        return slots
    spans = sorted((int(c["start_ms"]), int(c["end_ms"])) for c in locked)
    out: list[Slot] = []
    for s in slots:
        pieces = [(s.start_ms, s.end_ms)]
        for a, b in spans:
            nxt = []
            for x, y in pieces:
                if b <= x or y <= a:
                    nxt.append((x, y))
                    continue
                if x < a:
                    nxt.append((x, a))
                if b < y:
                    nxt.append((b, y))
            pieces = nxt
        for x, y in pieces:
            if y - x >= 120:
                frac = (y - x) / max(1, s.duration_ms)
                out.append(Slot(0, s.section_index, s.section_type, x, y,
                                max(1, int(round(s.beats * frac))), s.target))
            elif out and out[-1].end_ms == x:
                out[-1].end_ms = y   # sliver: absorb into the previous slot
    return out


def propose(analysis: dict, shots: list[dict], constraints: Constraints,
            *, query_scores: Optional[dict] = None, character_scene_ids: Iterable[int] = (),
            previous: Optional[list[dict]] = None, regenerate_sections: Optional[list[int]] = None,
            seed: Optional[int] = None) -> dict:
    """Build an edit proposal.

    ``shots``: dicts with scene_id, video_id, video_name, group, start_ms,
    end_ms and the Search+ metrics (intensity, activity, motion_type,
    motion_magnitude, flash_count; any may be None).
    ``query_scores``: {"global" | section_index: {scene_id: 0-1}}.
    ``previous`` + ``regenerate_sections``: keep the clips outside those
    sections; locked clips are always kept.
    """
    rng = random.Random(seed)
    query_scores = query_scores or {}
    warnings: list[str] = []
    chars = set(int(x) for x in character_scene_ids)

    pool = [dict(s) for s in shots]
    if constraints.groups:
        wanted = set(constraints.groups)
        pool = [s for s in pool if (s.get("group") or "__none__") in wanted]
        if not pool:
            warnings.append("no_shots_in_groups")
    ranks = _rank01([s.get("intensity") for s in pool])
    for s, r in zip(pool, ranks):
        s["_rank"] = r
        s["_character"] = int(s["scene_id"]) in chars
    by_id = {int(s["scene_id"]): s for s in pool}
    if constraints.characters and not chars:
        warnings.append("no_character_matches")

    previous = previous or []
    keep: list[dict] = []
    if previous:
        regen = set(regenerate_sections or [])
        for c in previous:
            if c.get("locked") or (regen and int(c["section_index"]) not in regen):
                keep.append(dict(c))
    if previous and regenerate_sections:
        # Same slot grid as before for the regenerated sections.
        slots = [Slot(0, int(c["section_index"]), _section_type(analysis, int(c["section_index"])),
                      int(c["start_ms"]), int(c["end_ms"]), int(c.get("beats") or 1),
                      _energy_at(analysis, int(c["start_ms"]), int(c["end_ms"])))
                 for c in previous
                 if not c.get("locked") and int(c["section_index"]) in set(regenerate_sections)]
    else:
        slots = _fit_locked(plan_slots(analysis, constraints.density), keep)

    usage = _Usage()
    for c in keep:
        usage.add(int(c["scene_id"]), int(c["src_in_ms"]), int(c["src_out_ms"]))

    target_ranks = _rank01([s.target for s in slots]) if slots else []
    # Calm passages ask for the calm end of the library even when the whole
    # song is loud: blend the song-relative rank with the absolute energy.
    target_ranks = [0.6 * r + 0.4 * s.target for r, s in zip(target_ranks, slots)]

    reused = 0
    unfilled = 0
    new_clips: list[dict] = []
    keep_sorted = sorted(keep, key=lambda c: int(c["start_ms"]))
    for slot, trank in zip(sorted(slots, key=lambda s: s.start_ms), target_ranks):
        length = slot.duration_ms
        prev_sid = _previous_scene(keep_sorted, new_clips, slot.start_ms)
        scored = []
        for shot in pool:
            if int(shot["end_ms"]) - int(shot["start_ms"]) < length:
                continue
            win = _window(shot, length, usage, constraints.avoid_reuse)
            if win is None:
                continue
            sc = _score(shot, slot, trank, query_scores, prev_sid, rng)
            if constraints.avoid_reuse:
                # Another moment of an already used shot still reads as a
                # repeat: only when the fresh shots are clearly worse.
                sc -= 0.35 * usage.count(int(shot["scene_id"]))
            scored.append((sc, shot, win))
        if not scored and constraints.avoid_reuse:
            # Library too small for this song: allow a shot again, least-used first.
            for shot in pool:
                if int(shot["end_ms"]) - int(shot["start_ms"]) < length:
                    continue
                win = _window(shot, length, usage, False)
                if win is None:
                    continue
                penalty = 0.4 * usage.count(int(shot["scene_id"]))
                scored.append((_score(shot, slot, trank, query_scores, prev_sid, rng) - penalty, shot, win))
            if scored:
                reused += 1
        if not scored:
            unfilled += 1
            continue
        scored.sort(key=lambda x: -x[0])
        best_score, best, win = scored[0]
        usage.add(int(best["scene_id"]), *win)
        clip = _clip(slot, best, win, best_score)
        seen = {int(best["scene_id"])}
        for sc, shot, w in scored[1:]:
            if int(shot["scene_id"]) in seen:
                continue
            seen.add(int(shot["scene_id"]))
            clip["alternatives"].append(_alt(shot, w, sc))
            if len(clip["alternatives"]) >= ALTERNATIVES:
                break
        new_clips.append(clip)

    clips = sorted(keep + new_clips, key=lambda c: int(c["start_ms"]))
    for i, c in enumerate(clips):
        c["slot"] = i
        shot = by_id.get(int(c["scene_id"]))
        if shot is not None and c.get("intensity") is None and shot.get("intensity") is not None:
            c["intensity"] = round(float(shot["intensity"]), 3)
    if reused:
        warnings.append(f"reused:{reused}")
    if unfilled:
        warnings.append(f"unfilled:{unfilled}")
    if not pool:
        warnings.append("empty_library")
    elif constraints.avoid_reuse and len({int(c["scene_id"]) for c in clips}) < len(clips):
        warnings.append(f"repeats:{len(clips) - len({int(c['scene_id']) for c in clips})}")
    return {"clips": clips, "warnings": warnings}


def _section_type(analysis: dict, index: int) -> str:
    for s in analysis.get("sections") or []:
        if int(s["index"]) == index:
            return s["type"]
    return "verse"


def _previous_scene(keep: list[dict], new: list[dict], start_ms: int) -> Optional[int]:
    best = None
    best_end = -1
    for c in list(keep) + list(new):
        end = int(c["end_ms"])
        if end <= start_ms and end > best_end:
            best, best_end = int(c["scene_id"]), end
    return best


def cut_offsets_ms(clips: list[dict], beats_ms: list[int], strong_ms: list[int]) -> list[int]:
    """Signed distance (ms) from each internal cut to the nearest beat or
    strong attack — the "do cuts land on the beat" check."""
    marks = np.asarray(sorted(set(int(b) for b in beats_ms) | set(int(s) for s in strong_ms)))
    out = []
    for c in clips[1:]:
        t = int(c["start_ms"])
        if marks.size:
            k = int(np.argmin(np.abs(marks - t)))
            out.append(int(t - marks[k]))
    return out
