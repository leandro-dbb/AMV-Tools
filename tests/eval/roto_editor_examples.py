"""The 3 example shots after the roto editor simulation (v06): what the AI
alone gives, what a few corrections give. Per example folder:

  detourage-alpha.mov    ProRes 4444 1080p of the corrected shot (the editor's export)
  comparaison.mp4        Original / AI alone / after corrections (black) / after
                         corrections (red), then a zoom x3 at half speed
  contour-sur-blanc.png  3 frames, zoom x3 on the outline: original / AI alone /
                         after corrections, on white
  contour-sur-rouge.png  the same on red
  chiffres.txt           gestures, time, metrics before / after

usage: python -m tests.eval.roto_editor_examples --sim <sim out> --out <reports dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tests.eval import roto_eval as E

R = Path("/srv/data/amv-tools/roto-stable/bench")
SRC = {"d1": "/srv/data/media/attente/[SubsPlease] Dragon Ball Super - Beerus - 01 (1080p) [B3FEF9B7].mkv",
       "d2": "/srv/data/media/attente/[SubsPlease] Dragon Ball Super - Beerus - 02 (1080p) [54BBBB45].mkv",
       "bc": "/srv/data/media/attente/Black.Clover.S02E04.Day.of.Atonement.1080p.BILI.WEB-DL.JPN.AAC2.0.H.265.MSubs-ToonsHub.mkv"}
EXAMPLES = [("b1", "01-calme-cigarette-black-clover"), ("s04", "02-quatre-personnages-plan-large"),
            ("s13", "03-aura-ssj3-camera-qui-tremble")]
W, H = 1920, 1080
CW, CH = 960, 540
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except Exception:
        return ImageFont.load_default()


def reader(args, ch, n):
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", *args, "-f", "rawvideo",
                          "-pix_fmt", "rgb24" if ch == 3 else "rgba", "pipe:1"], stdout=subprocess.PIPE)
    sz = W * H * ch
    for _ in range(n):
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, ch)
    p.stdout.close()
    p.wait()


def over(rgba, bg):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3] * a + bg * (1 - a)).clip(0, 255).astype(np.uint8)


def label(img, text, sz=30):
    im = Image.fromarray(np.ascontiguousarray(img))
    dr = ImageDraw.Draw(im)
    f = _font(sz)
    x0, y0, x1, y1 = dr.textbbox((14, 10), text, font=f)
    dr.rectangle((x0 - 8, y0 - 6, x1 + 8, y1 + 6), fill=(0, 0, 0))
    dr.text((14, 10), text, font=f, fill=(255, 255, 255))
    return np.asarray(im)


def fit(img, w, h):
    return np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((w, h), Image.LANCZOS))


def grid(cells, titles):
    g = np.zeros((H, W, 3), np.uint8)
    for k, (c, tt) in enumerate(zip(cells, titles)):
        r, q = divmod(k, 2)
        g[r * CH:(r + 1) * CH, q * CW:(q + 1) * CW] = label(fit(c, CW, CH), tt)
    return g


BLACK = np.zeros((H, W, 3), np.float32)
WHITE = np.full((H, W, 3), 255.0, np.float32)
RED = np.zeros((H, W, 3), np.float32)
RED[..., 0] = 220


def make_example(sid: str, dname: str, sim: Path, out: Path) -> None:
    from backend.indexing.cuts import probe_video
    from backend.roto.export import export_project
    from backend.roto.project import RotoProject
    res = json.loads((sim / f"{sid}.json").read_text())
    p = RotoProject.load(Path(res["project_dir"]))
    shot = {s["id"]: s for s in E.load_shots(R)}[sid]
    src = SRC[shot["src"]]
    fps = float(probe_video(src).get("fps") or 24.0)
    n = p.n
    start = int(round(shot["start"] * 1000))
    end = start + int(round(n * 1000 / fps))
    p.meta.update({"video_path": src, "start_ms": start, "end_ms": end, "fps": fps, "name": sid})
    d = out / dname
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    files = export_project(p, tmp, ["prores"], name="final")
    shutil.move(files["prores"], d / "detourage-alpha.mov")
    cur = p.masks["main"]
    p.masks["main"] = p.ai["main"]
    from backend.roto.edges import EdgeSettings
    saved = p.settings
    p.settings = EdgeSettings()   # the AI alone: default edge settings
    files = export_project(p, tmp, ["prores"], name="ai")
    ai_mov = Path(files["prores"])   # only for the comparison (kept out of the deliverables: size)
    p.masks["main"] = cur
    p.settings = saved

    def streams():
        orig = reader(["-ss", f"{start / 1000}", "-i", src, "-t", f"{(end - start) / 1000}", "-vf", f"fps={fps}",
                       "-frames:v", str(n)], 3, n)
        ai = reader(["-i", str(ai_mov)], 4, n)
        fin = reader(["-i", str(d / "detourage-alpha.mov")], 4, n)
        return zip(orig, ai, fin)

    mid = n // 2
    a_mid = None
    for i, (_o, _a, f) in enumerate(streams()):
        if i == mid:
            a_mid = f[..., 3].astype(np.float32) / 255.0
            break
    x0, y0, x1, y1 = E.hair_window(a_mid, size=(320, 180))
    titles = ["Original", "IA seule (passe automatique) — fond noir", "Après corrections — fond noir",
              "Après corrections — fond rouge"]
    enc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-framerate", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset",
                            "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y",
                            str(d / "comparaison.mp4")], stdin=subprocess.PIPE)
    for o, a, f in streams():
        enc.stdin.write(grid([o, over(a, BLACK), over(f, BLACK), over(f, RED)], titles).tobytes())
    zt = [titles[0] + " (zoom ×3, ralenti)"] + titles[1:]
    for o, a, f in streams():
        c = lambda im: im[y0:y1, x0:x1]   # noqa: E731
        g = grid([c(o), c(over(a, BLACK)), c(over(f, BLACK)), c(over(f, RED))], zt).tobytes()
        enc.stdin.write(g)
        enc.stdin.write(g)
    enc.stdin.close()
    enc.wait()

    # boards: 3 frames (start, middle, end), zoom x3 around the most detailed outline
    picks = sorted({n // 6, n // 2, (5 * n) // 6})
    rows = {}
    for i, (o, a, f) in enumerate(streams()):
        if i in picks:
            rows[i] = (o, a, f)
    for bgname, bg in (("blanc", WHITE), ("rouge", RED)):
        board = np.zeros((len(picks) * CH, 3 * CW, 3), np.uint8)
        for r, i in enumerate(picks):
            o, a, f = rows[i]
            wx0, wy0, wx1, wy1 = E.hair_window(f[..., 3].astype(np.float32) / 255.0, size=(320, 180))
            c = lambda im: fit(im[wy0:wy1, wx0:wx1], CW, CH)   # noqa: E731
            cells = [(c(o), f"Image {i} — original (zoom ×3)"), (c(over(a, bg)), "IA seule"),
                     (c(over(f, bg)), "Après corrections")]
            for q, (img, tt) in enumerate(cells):
                board[r * CH:(r + 1) * CH, q * CW:(q + 1) * CW] = label(img, tt, 26)
        Image.fromarray(board).save(d / f"contour-sur-{bgname}.png", optimize=True)
    shutil.rmtree(tmp, ignore_errors=True)

    ai, fin = res["ai"], res["final"]
    from tests.eval.roto_editor_sim import HUMAN_S_PER_GESTURE, QUIET_AI_S_PER_FRAME, left, quiet_s
    fmt = lambda v, d=1: "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (  # noqa: E731
        f"{v:.{d}f}" if isinstance(v, float) else str(v))
    lines = [
        f"Plan {sid} ({dname}) — {n} images, {fps:.3f} images/s",
        "",
        "Simulation d'un monteur (vérité terrain) : geste sur la plus grosse erreur de l'image la plus mauvaise,",
        "puis propagation dans les deux sens, jusqu'à « propre » ou 30 gestes.",
        "",
        f"Gestes                      : {res['gestures']} (dont {res['undos']} annulations), "
        f"{res['propagations']} propagations ; premier choix : "
        f"{'cliquer sur le personnage' if res.get('target') == 'click' else 'tous les personnages'}",
        f"Propre à la fin             : {'OUI' if res['final_clean']['all'] else 'NON'} "
        f"(critères : {', '.join(k for k, v in res['final_clean'].items() if k != 'all' and not v) or 'tous tenus'}"
        f"{' manqués' if not res['final_clean']['all'] else ''})",
        f"Temps machine, carte calme  : {quiet_s(res):.0f} s (passe IA {n * QUIET_AI_S_PER_FRAME:.0f} s, "
        f"propagations, gestes) — estimé à partir des vitesses mesurées seules sur la RTX 3090",
        f"Temps total estimé          : {quiet_s(res) + HUMAN_S_PER_GESTURE * res['gestures']:.0f} s "
        f"(+ {HUMAN_S_PER_GESTURE} s par geste pour viser et vérifier)",
        f"Mesuré pendant la simulation : {res['machine_s']:.0f} s (quatre simulations à la fois sur la carte : "
        f"clic {fmt(res['click_ms_med'], 0)} ms, pinceau {fmt(res['brush_ms_med'], 0)} ms en médiane)",
        "",
        f"Erreur restante (pixels faux sur les images de référence, en % de ceux de la passe IA) après",
        f"  5 / 10 / tous les gestes  : {left(res, ai)}",
        f"Recouvrement avec la référence (IoU) : {fmt(ai.get('J'), 3)} -> {fmt(fin.get('J'), 3)}",
        "",
        "                              IA seule   après corrections   cible",
        f"encre de contour (%)        : {fmt(ai['ink']):>8}   {fmt(fin['ink']):>8}            >= 97",
        f"fond le long du contour (%) : {fmt(ai['bg']):>8}   {fmt(fin['bg']):>8}            <= 3",
        f"rugosité du bord (%)        : {fmt(ai['jag']):>8}   {fmt(fin['jag']):>8}            <= {fmt(res['old_jag'])} (ancien mode)",
        f"trous (images de référence) : {fmt(ai['holes']):>8}   {fmt(fin['holes']):>8}            0",
        f"petits trous, toutes images : {fmt(ai.get('holes_all')):>8}   {fmt(fin.get('holes_all')):>8}            (vrais creux de fond compris)",
        f"images perdues              : {fmt(ai['lost']):>8}   {fmt(fin['lost']):>8}            0",
        f"objet tenu perdu (images)   : {fmt(ai['held_lost']):>8}   {fmt(fin['held_lost']):>8}            0",
        "",
        "Mesures sur les images ayant une référence (" + str(res["ref_frames"]) + "), alpha de l'export à la taille",
        "de travail ; images perdues sur toutes les images. Fichiers : detourage-alpha.mov (après",
        "corrections), comparaison.mp4, contour-sur-blanc.png, contour-sur-rouge.png.",
    ]
    (d / "chiffres.txt").write_text("\n".join(lines) + "\n")
    print("done", sid, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    for sid, dname in EXAMPLES:
        if a.only and sid not in a.only.split(","):
            continue
        make_example(sid, dname, Path(a.sim), Path(a.out))


if __name__ == "__main__":
    main()
