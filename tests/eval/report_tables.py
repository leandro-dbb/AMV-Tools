"""Markdown tables of the chantier 4 measurements (docs/reports/chantier-4-modeles.md).

    python tests/eval/report_tables.py WORK_DIR [--label NVIDIA]

Reads the *_runs.jsonl files written by eval_models.py / eval_finish.py and
the search/tag scores; one row per (model, simulated memory) — when a model
was measured several times at full size, the median and the spread are shown.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _runs(work: Path, name: str) -> list[dict]:
    p = work / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _key(r: dict) -> str:
    m = r["model"].replace("hf-hub:timm/", "").replace("google/", "").replace("SmilingWolf/", "")
    if r.get("patches") and "naflex" in r["model"]:
        m += f" p{r['patches']}"
    return m


def _med(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.median(vals), (max(vals) - min(vals)) if len(vals) > 1 else 0.0


def _fmt(v, nd=1):
    if v is None:
        return "–"
    return f"{v:.{nd}f}".replace(".", ",")


def table_speed(rows: list[dict], what: str) -> str:
    full = defaultdict(list)
    for r in rows:
        if r.get("sim_gb") or r.get("mps_ratio") or r.get("vram_cap_gb") or r.get("n", 0) < 1000:
            continue
        full[_key(r) + (r.get("suffix") or "")].append(r)
    out = [f"| {what} | débit (img/s) | chargement à froid (s) | mémoire GPU max (Go) | RAM max (Go) | runs |",
           "|---|---:|---:|---:|---:|---:|"]
    for k, rs in full.items():
        ok = [r for r in rs if r.get("ok")]
        if not ok:
            out.append(f"| {k} | échec : {rs[-1].get('error', '')[:60]} | | | | {len(rs)} |")
            continue
        ips, spread = _med([r.get("ips") for r in ok])
        load, _ = _med([r.get("load_s") for r in ok])
        mem, _ = _med([r.get("peak_reserved_gb") or r.get("peak_alloc_gb") for r in ok])
        ram, _ = _med([r.get("footprint_gb") or r.get("rss_peak_gb") for r in ok])
        sp = f" (±{_fmt(spread / 2)})" if spread else ""
        out.append(f"| {k} | {_fmt(ips)}{sp} | {_fmt(load)} | {_fmt(mem, 2)} | {_fmt(ram, 2)} | {len(ok)} |")
    return "\n".join(out)


def table_tiers(rows: list[dict], what: str) -> str:
    """pass / OOM per simulated memory size."""
    cells = defaultdict(dict)
    sizes = set()
    for r in rows:
        sim = r.get("sim_gb")
        if not sim:
            continue
        sizes.add(float(sim))
        st = "OK" if r.get("ok") else ("OOM" if r.get("oom") else "échec")
        mem = r.get("peak_reserved_gb") or r.get("peak_alloc_gb")
        cells[_key(r)][float(sim)] = f"{st} {_fmt(r.get('ips'))} img/s, {_fmt(mem, 2)} Go" if r.get("ok") else st
    sizes = sorted(sizes)
    out = [f"| {what} | " + " | ".join(f"{s:g} Go" for s in sizes) + " |",
           "|---|" + "---|" * len(sizes)]
    for k, c in cells.items():
        out.append(f"| {k} | " + " | ".join(c.get(s, "–") for s in sizes) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    a = ap.parse_args()
    w = Path(a.work)
    emb = _runs(w, "embed_runs.jsonl")
    tag = _runs(w, "tag_runs.jsonl")
    print("## embeddings — vitesse\n")
    print(table_speed(emb, "modèle"))
    print("\n## embeddings — paliers simulés\n")
    print(table_tiers(emb, "modèle"))
    print("\n## tagger — vitesse\n")
    print(table_speed(tag, "modèle"))
    print("\n## tagger — paliers simulés\n")
    print(table_tiers(tag, "modèle"))
    ss = w / "search_scores.json"
    if ss.exists():
        print("\n## recherche — qualité\n\n| modèle | P@10 | nDCG@10 | plans similaires P@5 | non jugés |\n|---|---:|---:|---:|---:|")
        for m, r in json.loads(ss.read_text()).items():
            print(f"| {m} | {_fmt(r['p10'], 3)} | {_fmt(r['ndcg10'], 3)} | {_fmt(r['similar_p5'], 3)} | {r['unjudged']} |")
    ts = w / "tag_scores.json"
    if ts.exists():
        print("\n## tagger — qualité\n\n| modèle | perso. P | perso. R | perso. F1 | tags P | tags R | tags F1 | personnages reconnus |\n|---|---:|---:|---:|---:|---:|---:|---|")
        for m, r in json.loads(ts.read_text()).items():
            c, t = r["characters"], r["tags"]
            print(f"| {m} | {_fmt(c['precision'], 2)} | {_fmt(c['recall'], 2)} | {_fmt(c['f1'], 2)} | "
                  f"{_fmt(t['precision'], 2)} | {_fmt(t['recall'], 2)} | {_fmt(t['f1'], 2)} | "
                  f"{', '.join(r['recognised_characters']) or '–'} |")
    for name, cols in (("upscale_runs.jsonl", ("fps_540_to_1080", "psnr", "ssim", "lpips", "psnr_dark")),
                       ("depth_runs.jsonl", ("fps_1080", "flicker_raw", "flicker_align", "flicker_app", "load_s"))):
        rs = _runs(w, name)
        if rs:
            print(f"\n## {name}\n\n| modèle | " + " | ".join(cols) + " | mémoire GPU (Go) |\n|---|" + "---:|" * (len(cols) + 1))
            for r in rs:
                label = r["model"] + (f" {r['input_size']}" if "input_size" in r else "")
                print(f"| {label} | " + " | ".join(_fmt(r.get(c), 3) for c in cols)
                      + f" | {_fmt(r.get('peak_reserved_gb') or r.get('peak_alloc_gb'), 2)} |")
    rs = _runs(w, "interp_runs.jsonl")
    if rs:
        print("\n## interpolation\n\n| modèle | im/s 1080p | calme PSNR | calme LPIPS | action PSNR | action SSIM | action LPIPS | doublons PSNR |\n|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rs:
            ci, ai, cd = r.get("calm_inbetween", {}), r.get("action_inbetween", {}), r.get("calm_dup", {})
            print(f"| {r['model']} | {_fmt(r.get('fps_1080'))} | {_fmt(ci.get('psnr'), 2)} | {_fmt(ci.get('lpips'), 4)} | "
                  f"{_fmt(ai.get('psnr'), 2)} | {_fmt(ai.get('ssim'), 4)} | {_fmt(ai.get('lpips'), 4)} | {_fmt(cd.get('psnr'), 2)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
