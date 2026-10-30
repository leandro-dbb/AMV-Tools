"""Roto editor: does the AI do 90 % of the work? (v06, measurement tool)

A simulated editor fixes each shot of the roto benchmark with the editor's
own tools (backend/roto), guided by the ground truth:

  1. the AI pass is the Stable v5 Auto pipeline (``--ai cached``: its run on
     the benchmark, <root>/runs/app_stable; ``--ai live``: run now);
  2. loop: take the worst frame (a frame with a reference, or a frame the
     matte lost), the biggest error on it (a missing or extra piece,
     compared with the reference), and fix it with ONE gesture — a click
     (positive on a missing piece, negative on an extra one; SAM 2.1) when
     the piece is big and compact, a brush stroke otherwise (the piece's
     medial axis, brush radius = its median half-thickness, as a user who
     zooms in would paint); a click that makes things worse is undone (one
     more gesture) and replaced by a stroke; then propagate both ways
     (SAM 2.1 video predictor from the corrected frames) — and start again;
  3. stop when the shot is "clean" or after ``--max-gestures``.

"Clean" (the brief): contour ink kept >= 97 %, 0 lost frame, 0 hole,
roughness <= the old 0.4 mode on the same shot, background along the
contour <= 3 % — measured like the benchmark (tests/eval/roto_eval.py) on
the frames with a reference, after the export's edge finishing at the
working size (Stable v5 "soft"); lost frames and holes on every frame.
For b1, the cigarette is also checked on its 84 frames (< 50 % = lost).

Run on the GPU machine::

    python -m tests.eval.roto_editor_sim run --root <bench> --out <dir> --shots b1,s04,s13
    python -m tests.eval.roto_editor_sim latency --root <bench> --shot b1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from tests.eval import roto_eval as RE

INK_MIN = 97.0
BG_MAX = 3.0
CLICK_MIN_PX = 600       # pieces at least this big ...
CLICK_MIN_R = 7.0        # ... and this thick (px) are clicked; smaller ones painted
CLEAR_FRAC = 0.4         # a wrong piece over this share of the frame: "Empty this frame"
CLICK_MIN_GAIN = 0.25    # a click that fixes less than this share of the piece is undone, then painted
MIN_PIECE = 15           # error pieces under this many pixels are not worth a gesture
INK_MIN_PX = 30          # stretches of ink line shorter than this are not worth a stroke
LOST_FRAC_REF = 0.6      # a lost frame next to a reference this much smaller than usual is justified
TUNE_BELOW_PX = 120     # the edge sliders are tried once the pieces left are this small
HUMAN_S_PER_GESTURE = 5  # rough time a user spends per gesture (look, aim, check), for the total


class SimState:
    """What backend.roto.engine needs from the app state, without the app."""

    def __init__(self, dev):
        self.device = dev
        self.settings = {"models": {}}
        self._sam2 = {}

    def get_sam2(self, variant=None):
        from backend.models.sam2 import SAM2Model
        v = variant or "large"
        if v not in self._sam2:
            self._sam2[v] = SAM2Model(self.device, variant=v, idle_offload_seconds=0)
        return self._sam2[v]

    def get_birefnet(self, variant=None):
        raise RuntimeError("not used")

    def model_selection(self):
        return {"platform": "nvidia", "effective_tier": "max"}


# ── metrics ────────────────────────────────────────────────────────────────
_EXP: dict = {}


def exported(project, t: int) -> np.ndarray:
    """Frame t's matte as the export writes it (edge settings, ink line
    added, tiny holes closed: the alpha of edge_refine.finish_edges "soft"),
    at the working size; cached per frame version and settings."""
    from backend.roto.edges import apply_edges
    ver = (project.frame_versions.get(t, 0), project.settings.key())
    hit = _EXP.get((project.id, t))
    if hit is not None and hit[0] == ver:
        return hit[1]
    a = apply_edges(project.matte("main", t), project.rgb(t), project.settings)
    _EXP[(project.id, t)] = (ver, a)   # one per frame: bounded memory
    return a


def hole_pieces(a: np.ndarray):
    """Small transparent holes enclosed in the matte (roto_eval.holes), as masks."""
    import cv2
    inv = (a < 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = a.shape
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    return [lab == i for i in range(1, n) if i not in border and st[i, 4] < 0.001 * h * w]


def real_hole(piece: np.ndarray, gt) -> bool:
    """A hole the reference also has (background between strands): right."""
    return gt is not None and float(gt[piece].mean()) < 0.5


def frame_metrics(rgb, a_exp, gt) -> dict:
    m = RE.bg_metrics(a_exp, gt)
    return {"ink": RE.ink_kept(rgb, a_exp, gt), "bg": m["bg_band"], "eaten": m["eaten"],
            "jag": RE.jaggies(a_exp)}


def shot_metrics(project, gts: dict, cuts, held: dict, all_frames: bool = True) -> dict:
    from backend.models.auto_roto import lost_frames
    per = {}
    err, js = 0.0, []
    for t, gt in gts.items():
        per[t] = frame_metrics(project.rgb(t), exported(project, t), gt)
        a = project.matte("main", t)
        fn, fp = _error_maps(a, gt)
        err += float(fn.sum() + fp.sum())
        u = ((a > 0.5) | gt).sum()
        js.append(float(((a > 0.5) & gt).sum() / u) if u else 1.0)
    masks = project.masks["main"]
    lost = lost_frames(masks.astype(np.float32) / 255.0, cuts)
    # a frame "lost" where the reference itself has (almost) nothing — smoke
    # hiding the character (s16), an exit — is right, not an error: the
    # nearest reference within 8 frames tells
    areas = {u: float(g.sum()) for u, g in gts.items()}
    med = float(np.median(list(areas.values()))) if areas else 0.0
    lost_ok = []
    for t in lost:
        near = sorted((abs(u - t), u) for u in areas if abs(u - t) <= 8)
        if near and areas[near[0][1]] < LOST_FRAC_REF * med:
            lost_ok.append(t)
    lost = [t for t in lost if t not in lost_ok]
    # holes: only where the reference can tell a missing piece of the
    # character from a real pocket of background (between an arm and the
    # body); every small enclosed hole of the other frames is counted apart
    holes, hole_frames, holes_all = 0, [], 0
    for t in range(project.n):
        pcs = hole_pieces(exported(project, t))
        holes_all += len(pcs)
        if t in gts:
            k = sum(1 for pc in pcs if not real_hole(pc, gts[t]))
            holes += k
            if k:
                hole_frames.append(t)
    held_lost = 0
    held_min = float("nan")
    if held:
        vals = [RE.held_kept(masks[t].astype(np.float32) / 255.0, p) for t, p in held.items()]
        vals = [v for v in vals if np.isfinite(v)]
        held_lost = int(sum(v < 50 for v in vals))
        held_min = float(min(vals)) if vals else float("nan")
    nm = lambda k: float(np.nanmean([v[k] for v in per.values()])) if per else float("nan")   # noqa: E731
    return {"ink": nm("ink"), "ink_min": float(np.nanmin([v["ink"] for v in per.values()])) if per else float("nan"),
            "bg": nm("bg"), "eaten": nm("eaten"), "jag": nm("jag"), "holes_all": holes_all, "holes": holes,
            "hole_frames": hole_frames, "lost": len(lost), "lost_ids": lost, "lost_ok": lost_ok, "held_lost": held_lost,
            "held_min": held_min, "err": err, "J": float(np.mean(js)) if js else float("nan"), "per": per}


def is_clean(m: dict, old_jag: float) -> dict:
    ok = {"ink": m["ink"] >= INK_MIN, "lost": m["lost"] == 0, "holes": m["holes"] == 0,
          "rough": not np.isfinite(old_jag) or m["jag"] <= old_jag, "bg": m["bg"] <= BG_MAX,
          "held": m["held_lost"] == 0}
    ok["all"] = all(ok.values())
    return ok


# ── the simulated user ─────────────────────────────────────────────────────
def _error_maps(a: np.ndarray, gt: np.ndarray):
    import cv2
    g = gt.astype(np.uint8)
    k = np.ones((5, 5), np.uint8)   # 2 px: the reference is only that precise
    core = cv2.erode(g, k) > 0
    out = cv2.dilate(g, k) == 0
    fn = core & (a < 0.5)
    fp = out & (a > 0.3)
    return fn, fp


def frame_error(project, t, gt) -> float:
    if gt is None:
        return float("nan")
    fn, fp = _error_maps(project.matte("main", t), gt)
    return float(fn.sum() + fp.sum())


def biggest_piece(fn: np.ndarray, fp: np.ndarray):
    import cv2
    best = None
    for kind, m in (("add", fn), ("remove", fp)):
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if st[i, 4] >= MIN_PIECE and (best is None or st[i, 4] > best[2]):
                best = (kind, lab == i, int(st[i, 4]))
    return best


def pick_target(p, gts: dict, m: dict, c: dict, stuck: set):
    """The next thing a user would fix: a lost frame; else the biggest
    piece of error on any frame (missing / extra piece against the
    reference, a false hole, a stretch of ink line not kept)."""
    from backend.models.edge_refine import outline_ink
    import cv2
    for t in m["lost_ids"]:
        if (t, "lost") in stuck or t in m.get("lost_ok", ()):
            continue
        stuck.add((t, "lost"))
        nb = p.matte("main", t - 1 if t > 0 else t + 1) > 0.5
        if nb.any():
            x, y, _r = click_point(nb)
            return ("click_lost", t, x, y)
    cands = []
    for t, gt in gts.items():
        if (t, "piece") in stuck:
            continue
        fn, fp = _error_maps(p.matte("main", t), gt)
        pc = biggest_piece(fn, fp)
        if pc is not None:
            cands.append((pc[2], ("piece", t, pc[0], pc[1])))
    if not c["holes"]:
        for t in m["hole_frames"]:
            for pc in hole_pieces(exported(p, t)):
                if real_hole(pc, gts.get(t)):
                    continue
                key = (t, "hole", int(pc.sum()), int(np.argmax(pc)))
                if key in stuck:
                    continue
                cands.append((int(pc.sum()), ("hole", t, "add", pc, int(np.argmax(pc)))))
    if not c["ink"]:
        for t, gt in gts.items():
            ink = outline_ink(p.rgb(t), gt, 4) & (exported(p, t) < 0.9)
            n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                if st[i, 4] < INK_MIN_PX:
                    continue
                pc = lab == i
                key = (t, "ink", int(st[i, 4]), int(np.argmax(pc)))
                if key in stuck:
                    continue
                cands.append((int(st[i, 4]), ("ink", t, "add", pc, int(np.argmax(pc)))))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    return cands[0][1]


def stroke_for(piece: np.ndarray, min_r: float = 1.5):
    """One brush stroke over a piece: points along its medial axis (ridge of
    the distance transform), greedy nearest-neighbour order, radius = the
    median half-thickness there."""
    import cv2
    d = cv2.distanceTransform(piece.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = (d >= cv2.dilate(d, np.ones((3, 3), np.uint8)) - 1e-3) & piece
    ys, xs = np.nonzero(ridge)
    if len(xs) == 0:
        ys, xs = np.nonzero(piece)
    r = float(np.clip(np.median(d[ys, xs]) if len(xs) else 2.0, min_r, 40.0))
    pts = list(zip(xs.tolist(), ys.tolist()))
    step = max(1, int(r // 2))
    pts = pts[::step] if len(pts) > 400 else pts
    order = [pts.pop(0)]
    while pts:
        x0, y0 = order[-1]
        j = int(np.argmin([(x - x0) ** 2 + (y - y0) ** 2 for x, y in pts]))
        order.append(pts.pop(j))
    strokes, cur = [], [order[0]]
    for p in order[1:]:   # a jump far away starts a new segment of the same scribble
        if (p[0] - cur[-1][0]) ** 2 + (p[1] - cur[-1][1]) ** 2 > (3 * r) ** 2:
            strokes.append(cur)
            cur = []
        cur.append(p)
    strokes.append(cur)
    return strokes, r


def click_point(piece: np.ndarray):
    import cv2
    d = cv2.distanceTransform(piece.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(x), float(y), float(d[y, x])


def simulate_shot(sid: str, root: Path, out: Path, state, *, ai: str = "cached", max_gestures: int = 30,
                  log=print) -> dict:
    from backend.roto import engine
    from backend.roto.project import RotoProject
    paths = RE.frame_paths(root, sid)
    n = len(paths)
    proj_root = out / "projects"
    shutil.rmtree(proj_root / sid, ignore_errors=True)
    p = RotoProject.create(paths, fps=24.0, name=sid, root=proj_root)
    cuts = RE.shot_cuts(root, sid)
    p.meta["shots"] = cuts
    gts = {int(f.stem): RE.read_alpha(f) > 0.5 for f in sorted((root / "gt" / sid).glob("*.png"))}
    held = {int(f.stem): f for f in sorted((root / "gt_held_seq" / sid).glob("*.png"))} \
        if (root / "gt_held_seq" / sid).exists() else {}
    old = json.loads((root / "sc_old_jag.json").read_text())["birefnet@v05"].get(sid, {})
    old_jag = float(old.get("jag", float("nan")))
    t0 = time.perf_counter()
    rd = root / "runs" / "app_stable" / sid
    masks = np.stack([RE.read_alpha(rd / f"{i:05d}.png") for i in range(n)])
    st = json.loads((rd / "stats.json").read_text()) if (rd / "stats.json").exists() else {}
    ai_s = float(st.get("ms_per_frame", 1100.0)) * n / 1000.0
    first_target = "all"
    seed = RE.seed_frame(root, sid, n)
    if seed in gts and ai != "cached":
        # the user's first choice: "all the characters", or "click on the
        # character" when the AI takes characters the shot is not about
        # (a crowd behind Beerus): extra detected pieces away from the
        # reference, over a third of the reference's area
        import cv2
        g = gts[seed]
        near = cv2.dilate(g.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        extra = float(((masks[seed] > 0.5) & ~near).sum())
        if extra > 0.33 * float(g.sum()):
            first_target = "click"
            from backend.state import get_state
            pts = RE.click_points(g)
            t1 = time.perf_counter()
            masks = engine.ai_pass(p, get_state(), target="click", frame=seed, positive=pts)
            ai_s = time.perf_counter() - t1
            log(f"[{sid}] first pass: click on the character(s) {pts} on frame {seed} ({ai_s:.0f} s)")
    p.apply("main", {t: masks[t] for t in range(n)}, label="ai", ai=True, record=False)
    m0 = shot_metrics(p, gts, cuts, held)
    c0 = is_clean(m0, old_jag)
    log(f"[{sid}] AI pass: ink {m0['ink']:.1f} bg {m0['bg']:.1f} jag {m0['jag']:.1f} (old {old_jag:.1f}) "
        f"holes {m0['holes']}/{m0['holes_all']} lost {m0['lost']} held_lost {m0['held_lost']} -> {c0}")
    gestures, props, undos, ops = 0, 0, 0, []
    gesture_ms, click_ms, brush_ms, prop_s = [], [], [], []
    m, c = m0, c0
    stuck: set = set()
    tuned = False
    while not c["all"] and gestures < max_gestures:
        target = pick_target(p, gts, m, c, stuck)
        tsize = 10 ** 9 if target is None or target[0] == "click_lost" else int(target[3].sum())
        edge_fail = not (c["ink"] and c["bg"] and c["rough"])
        if edge_fail and not tuned and (target is None or tsize < TUNE_BELOW_PX):
            # what is left is along the whole edge: the edge sliders, as a user would try them
            tuned = True
            g0 = time.perf_counter()

            def score(mm, cc):
                return (sum(cc[k] for k in ("ink", "bg", "rough")),
                        min(mm["ink"] - INK_MIN, 0) - 3 * max(mm["bg"] - BG_MAX, 0))
            best = (score(m, c), p.settings.grow_px, p.settings.softness)
            for attr, values in (("grow_px", (0.5, 1.0, -0.5)), ("softness", (-0.4, 0.3))):
                keep = getattr(p.settings, attr)
                for v in values:
                    setattr(p.settings, attr, v)
                    gestures += 1
                    mm = shot_metrics(p, gts, cuts, held)
                    sc = score(mm, is_clean(mm, old_jag))
                    if sc > best[0]:
                        best = (sc, p.settings.grow_px, p.settings.softness)
                        keep = v
                setattr(p.settings, attr, keep)
            p.settings.grow_px, p.settings.softness = best[1], best[2]
            gesture_ms.append((time.perf_counter() - g0) * 1000)
            m = shot_metrics(p, gts, cuts, held)
            c = is_clean(m, old_jag)
            ops.append({"gesture": gestures, "frame": -1,
                        "kind": f"edge sliders: grow {best[1]} px, softness {best[2]}",
                        "ink": round(m["ink"], 2), "bg": round(m["bg"], 2), "jag": round(m["jag"], 2),
                        "holes": m["holes"], "lost": m["lost"], "err": m["err"], "J": round(m["J"], 4), "clean": c})
            log(f"[{sid}] g{gestures} edge sliders -> grow {best[1]} softness {best[2]} | ink {m['ink']:.1f} "
                f"bg {m['bg']:.2f} jag {m['jag']:.1f} holes {m['holes']} {c}")
            continue
        if target is None:
            log(f"[{sid}] nothing left to fix with a gesture (criteria: {c})")
            break
        kind_t, t = target[0], target[1]
        gt = gts.get(t)
        before = frame_error(p, t, gt) if gt is not None else None
        g0 = time.perf_counter()
        did = None
        if kind_t == "click_lost":
            region, _mod = engine.click_region(p, state, t, target[2], target[3])
            new = engine.apply_click(p.matte("main", t), region, True)
            p.apply("main", {t: new}, label="click", lock=[t])
            click_ms.append((time.perf_counter() - g0) * 1000)
            gestures += 1
            did = "click+ (lost frame)"
        else:
            mode, piece = target[2], target[3]
            area = int(piece.sum())
            x, y, r = click_point(piece)
            if kind_t == "piece" and mode == "remove" and area > CLEAR_FRAC * piece.size:
                # the AI took (almost) the whole frame, the character is not
                # there (smoke): "Empty this frame", one gesture
                h_, w_ = piece.shape
                new = engine.paint(p.matte("main", t), None, [{
                    "points": [(0, 0), (w_, 0), (w_, h_), (0, h_), (0, 0), (w_, h_)],
                    "radius": min(512, max(w_, h_) / 2), "mode": "remove"}], rgb=p.rgb(t))
                p.apply("main", {t: new}, label="brush", lock=[t])
                gestures += 1
                did = "empty frame"
            elif kind_t == "piece" and area >= CLICK_MIN_PX and r >= CLICK_MIN_R:
                region, _mod = engine.click_region(p, state, t, x, y, mode == "add")
                new = engine.apply_click(p.matte("main", t), region, mode == "add")
                p.apply("main", {t: new}, label="click", lock=[t])
                click_ms.append((time.perf_counter() - g0) * 1000)
                gestures += 1
                did = "click+" if mode == "add" else "click-"
                if frame_error(p, t, gt) > before - CLICK_MIN_GAIN * area:   # little gain: undo, then paint
                    p.undo()
                    gestures += 1
                    undos += 1
                    did = None
            if did is None:
                g1 = time.perf_counter()
                strokes, rad = stroke_for(piece, min_r=1.5 if kind_t != "hole" else 2.0)
                ai_t = p.ai["main"][t].astype(np.float32) / 255.0
                new = engine.paint(p.matte("main", t), ai_t,
                                   [{"points": s_, "radius": rad, "mode": mode} for s_ in strokes], rgb=p.rgb(t))
                p.apply("main", {t: new}, label="brush", lock=[t])
                brush_ms.append((time.perf_counter() - g1) * 1000)
                gestures += 1
                did = f"brush{'+' if mode == 'add' else '-'} {kind_t} r{rad:.1f}"
        gesture_ms.append((time.perf_counter() - g0) * 1000)
        after = frame_error(p, t, gt) if gt is not None else None
        if after is not None and before is not None and after >= before and kind_t == "piece":
            stuck.add((t, "piece"))
        if kind_t in ("hole", "ink"):
            stuck.add((t, kind_t, int(target[3].sum()), target[4]))
        # propagate both ways from the corrected frame
        q0 = time.perf_counter()
        new, method = engine.propagate(p, state, "main", t, "both")
        new = {u: mm for u, mm in new.items() if u not in p.locked["main"]}
        if new:
            p.apply("main", new, label="propagate")
        props += 1
        prop_s.append(time.perf_counter() - q0)
        m = shot_metrics(p, gts, cuts, held)
        c = is_clean(m, old_jag)
        ops.append({"gesture": gestures, "frame": t, "kind": did, "err_before": before, "err_after": after,
                    "propagated": len(new), "prop_s": round(prop_s[-1], 2), "method": method,
                    "ink": round(m["ink"], 2), "bg": round(m["bg"], 2), "jag": round(m["jag"], 2),
                    "holes": m["holes"], "lost": m["lost"], "err": m["err"], "J": round(m["J"], 4), "clean": c})
        log(f"[{sid}] g{gestures} f{t} {did}: err {before}->{after}, prop {len(new)} fr {prop_s[-1]:.1f}s | "
            f"ink {m['ink']:.1f} bg {m['bg']:.2f} jag {m['jag']:.1f} holes {m['holes']} lost {m['lost']} {c}")
    machine_s = ai_s + sum(gesture_ms) / 1000 + sum(prop_s)
    p.save()
    res = {
        "shot": sid, "frames": n, "ref_frames": len(gts), "old_jag": old_jag, "target": first_target,
        "ai": {k: v for k, v in m0.items() if k != "per"}, "ai_clean": c0,
        "final": {k: v for k, v in m.items() if k != "per"}, "final_clean": c,
        "gestures": gestures, "undos": undos, "propagations": props,
        "click_ms_med": float(np.median(click_ms)) if click_ms else None,
        "brush_ms_med": float(np.median(brush_ms)) if brush_ms else None,
        "prop_s_med": float(np.median(prop_s)) if prop_s else None,
        "ai_s": ai_s, "machine_s": machine_s,
        "total_s_with_user": machine_s + HUMAN_S_PER_GESTURE * gestures, "ops": ops,
        "project_dir": str(p.dir),
    }
    (out / f"{sid}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def cmd_run(args):
    dev = RE._device()
    state = SimState(dev)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for sid in args.shots.split(","):
        allres[sid] = simulate_shot(sid, root, out, state, ai=args.ai, max_gestures=args.max_gestures)
    (out / "summary.json").write_text(json.dumps(allres, indent=1, default=float))


def cmd_latency(args):
    """Live correction speed on one frame: click (image embedding + mask),
    brush, matte preview with the edge settings; propagation per frame."""
    from backend.roto import engine
    from backend.roto.edges import apply_edges
    from backend.roto.project import RotoProject
    import torch
    dev = RE._device()
    state = SimState(dev)
    root = Path(args.root)
    paths = RE.frame_paths(root, args.shot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "lat", ignore_errors=True)
    p = RotoProject.create(paths, fps=24.0, name=args.shot, root=out / "lat")
    rd = root / "runs" / "app_stable" / args.shot
    p.apply("main", {t: RE.read_alpha(rd / f"{t:05d}.png") for t in range(p.n)}, label="ai", ai=True, record=False)
    res = {"frame_size": p.size, "device": str(dev.torch_device)}

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    t = p.n // 2
    m = p.matte("main", t) > 0.5
    x, y, _ = click_point(m)
    engine.click_region(p, state, 0, x, y)   # warm-up (model load)
    lat = {"click_new_frame": [], "click_same_frame": [], "brush": [], "preview": []}
    for k in range(5):
        u = (t + 7 * k) % p.n
        sync(); a = time.perf_counter()
        region, model = engine.click_region(p, state, u, x, y)
        new = engine.apply_click(p.matte("main", u), region, True)
        p.apply("main", {u: new}, label="click", lock=[u])
        sync(); lat["click_new_frame"].append((time.perf_counter() - a) * 1000)
        a = time.perf_counter()
        region, model = engine.click_region(p, state, u, x + 5, y + 5)
        new = engine.apply_click(p.matte("main", u), region, False)
        p.apply("main", {u: new}, label="click", lock=[u])
        sync(); lat["click_same_frame"].append((time.perf_counter() - a) * 1000)
        a = time.perf_counter()
        new = engine.paint(p.matte("main", u), None, [{"points": [(x, y), (x + 40, y + 10), (x + 80, y)],
                                                         "radius": 14, "mode": "add"}])
        p.apply("main", {u: new}, label="brush", lock=[u])
        lat["brush"].append((time.perf_counter() - a) * 1000)
        a = time.perf_counter()
        apply_edges(p.matte("main", u), p.rgb(u), p.settings)
        lat["preview"].append((time.perf_counter() - a) * 1000)
    res.update({k: {"median_ms": float(np.median(v)), "all": [round(x_, 1) for x_ in v]} for k, v in lat.items()})
    res["click_model"] = model
    a = time.perf_counter()
    new, method = engine.propagate(p, state, "main", t, "both")
    sync()
    dt = time.perf_counter() - a
    res["propagate"] = {"frames": len(new), "s": dt, "ms_per_frame": dt * 1000 / max(1, len(new)), "method": method}
    if torch.cuda.is_available():
        res["vram_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    print(json.dumps(res, indent=1))
    (out / f"latency_{args.shot}.json").write_text(json.dumps(res, indent=1))


QUIET_AI_S_PER_FRAME = 1.1     # Stable v5 AI pass on a quiet RTX 3090 (chantier roto, part 5 §8)
QUIET_PROP_S_PER_FRAME = 0.18  # SAM 2.1 large propagation, quiet (latency run)
QUIET_GESTURE_S = 0.15         # click / brush, quiet


def quiet_s(r: dict) -> float:
    """Machine time of a simulated session on a quiet RTX 3090: the
    simulations ran four at a time, which inflates their measured times."""
    props = sum(int(o.get("propagated", 0)) for o in r.get("ops", []))
    return (r["frames"] * QUIET_AI_S_PER_FRAME + props * QUIET_PROP_S_PER_FRAME
            + r["gestures"] * QUIET_GESTURE_S)


def left(r: dict, a: dict) -> str:
    """Error left (% of the AI pass's) after 5 / 10 gestures and at the end."""
    base = a.get("err")
    if not base:
        return "–"
    out = []
    for k in (5, 10):
        ops = [o for o in r.get("ops", []) if o.get("err") is not None and o["gesture"] <= k]
        out.append(f"{100 * ops[-1]['err'] / base:.0f}" if ops else "–")
    out.append(f"{100 * r['final']['err'] / base:.0f}" if r["final"].get("err") is not None else "–")
    return " / ".join(out) + " %"


def cmd_table(args):
    """Markdown table of a simulation folder (one row per shot)."""
    out = Path(args.out)
    from backend.roto.project import RotoProject
    root = Path(args.root)
    rows = []
    for f in sorted(out.glob("*.json")):
        if f.name == "summary.json":
            continue
        r = json.loads(f.read_text())
        # overlap with the reference (IoU) and error pixels, AI pass vs final, from the saved project
        try:
            p = RotoProject.load(Path(r["project_dir"]))
            gts = {int(g.stem): RE.read_alpha(g) > 0.5 for g in sorted((root / "gt" / r["shot"]).glob("*.png"))}
            for key, get in (("ai", lambda t: p.ai["main"][t] / 255.0), ("final", lambda t: p.matte("main", t))):
                js, es = [], 0.0
                for t, g in gts.items():
                    a = get(t) > 0.5
                    u = (a | g).sum()
                    js.append((a & g).sum() / u if u else 1.0)
                    fn, fp = _error_maps(get(t), g)
                    es += fn.sum() + fp.sum()
                r[key]["J"] = float(np.mean(js))
                r[key]["err"] = float(es)
        except Exception as exc:   # noqa: BLE001
            print("no project for", r["shot"], exc, file=sys.stderr)
        rows.append(r)
    fm = lambda v, d=1: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"   # noqa: E731
    lines = ["| plan | images (réf.) | 1er choix | gestes | propagations | IoU IA → fin | erreur restante après 5 / 10 / tous les gestes | encre % IA → fin | fond contour % IA → fin | "
             "rugosité % IA → fin (ancien) | trous IA → fin | perdues IA → fin | propre ? (critères manqués) | "
             "temps estimé machine calme + 5 s/geste |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, z, c = r["ai"], r["final"], r["final_clean"]
        miss = [k for k, v in c.items() if k != "all" and not v]
        lines.append(
            f"| {r['shot']} | {r['frames']} ({r['ref_frames']}) | {r.get('target', 'all')} | {r['gestures']}"
            f"{' (' + str(r['undos']) + ' annul.)' if r['undos'] else ''} | {r['propagations']} | "
            f"{fm(a.get('J'), 3)} → {fm(z.get('J'), 3)} | "
            f"{left(r, a)} | "
            f"{fm(a['ink'])} → {fm(z['ink'])} | {fm(a['bg'])} → {fm(z['bg'])} | {fm(a['jag'])} → {fm(z['jag'])} "
            f"({fm(r['old_jag'])}) | {a['holes']} → {z['holes']} | {a['lost']} → {z['lost']} | "
            f"{'oui' if c['all'] else 'non (' + ', '.join(miss) + ')'} | {quiet_s(r) / 60:.1f} + "
            f"{HUMAN_S_PER_GESTURE * r['gestures'] / 60:.1f} min |")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shots", required=True)
    r.add_argument("--ai", default="choose", help="cached: always the cached 'all' pass; choose: like a user")
    r.add_argument("--max-gestures", type=int, default=30)
    r.set_defaults(fn=cmd_run)
    la = sub.add_parser("latency")
    la.add_argument("--root", required=True)
    la.add_argument("--out", required=True)
    la.add_argument("--shot", default="b1")
    la.set_defaults(fn=cmd_latency)
    tb = sub.add_parser("table")
    tb.add_argument("--out", required=True)
    tb.add_argument("--root", default="/srv/data/amv-tools/roto-stable/bench")
    tb.set_defaults(fn=cmd_table)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
