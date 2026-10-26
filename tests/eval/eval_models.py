"""Model evaluation harness for chantier 4 (best model per VRAM tier).

Not a unit test: run by hand on the measurement machines. Every sub-command
writes plain JSON/NPZ next to ``--work`` so the hand judgments in this folder
(``search_judgments.json``, ``tagger_truth.json``) can be re-scored against
new models without redoing anything.

    # 1. median frames (512x288, the frames the indexing pipeline feeds the
    #    models) of every scene of an indexed bench database
    python tests/eval/eval_models.py dump --db DATA/amv_tools.db --work W
    # 2. embeddings + timings of one model (optionally under a VRAM cap)
    python tests/eval/eval_models.py embed --work W --model google/siglip2-so400m-patch16-naflex
    # 3. tagger scores + timings
    python tests/eval/eval_models.py tag --work W --model SmilingWolf/wd-eva02-large-tagger-v3
    # 4. scores (P@10, nDCG@10, tagger P/R/F1) from the judgment files
    python tests/eval/eval_models.py score-search --work W
    python tests/eval/eval_models.py score-tags --work W
    # contact sheets for judging (pooled top-10 of every model per query)
    python tests/eval/eval_models.py sheets --work W --out SHEETS

Timings: ``load_s`` = process-cold model load (weights in the OS page cache,
as on the second launch of the app), ``ips`` = images/s over the whole set
after one warm-up batch, ``peak_alloc_gb``/``peak_reserved_gb`` = torch
allocator peaks (CUDA) or Metal driver allocation (MPS), ``rss_peak_gb`` =
process max RSS. ``--vram-gb N`` caps the torch allocator to N GB
(``set_per_process_memory_fraction`` on CUDA, the high watermark on MPS) and
records OOM instead of crashing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

HERE = Path(__file__).resolve().parent


# ── helpers ─────────────────────────────────────────────────────────────────
def _backend() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cap_memory(vram_gb: float | None, backend: str) -> None:
    if not vram_gb:
        return
    import torch
    if backend == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb / total))


class _MemSampler:
    """Peak Metal allocation on MPS (no allocator peak counter there)."""

    def __init__(self, backend: str):
        self.backend, self.peak = backend, 0
        self._stop = threading.Event()
        self._t = None

    def start(self):
        if self.backend != "mps":
            return self
        import torch

        def run():
            while not self._stop.wait(0.05):
                try:
                    self.peak = max(self.peak, torch.mps.driver_allocated_memory())
                except Exception:
                    pass
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join()


def _sync(backend: str) -> None:
    import torch
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()


def _mem_report(backend: str, sampler: _MemSampler) -> dict:
    import torch
    out = {"rss_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                / (2**30 if sys.platform == "darwin" else 2**20), 2)}
    if backend == "cuda":
        out["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        out["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    elif backend == "mps":
        out["peak_alloc_gb"] = round(sampler.peak / 2**30, 2)
    if sys.platform == "darwin":
        try:
            sys.path.insert(0, str(_REPO / "tests"))
            from bench_pipeline import _footprint
            fp = _footprint(os.getpid())
            out["footprint_gb"] = round(fp / 2**30, 2) if fp else None
        except Exception:
            pass
    return out


def _is_oom(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "out of memory" in s or "mps backend out of memory" in s or isinstance(exc, MemoryError)


def _load_frames(work: Path):
    from PIL import Image
    man = json.loads((work / "manifest.json").read_text())
    imgs = [Image.open(work / "frames" / f"{s['key']}.png").convert("RGB") for s in man["scenes"]]
    return man, imgs


# ── dump ────────────────────────────────────────────────────────────────────
def cmd_dump(args) -> None:
    from backend.indexing import frames as fr

    work = Path(args.work)
    (work / "frames").mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        vids = conn.execute("SELECT id, filepath, display_name FROM videos ORDER BY id").fetchall()
        rows = conn.execute("SELECT id, video_id, scene_index, start_ms, end_ms, parent_scene_id "
                            "FROM scenes ORDER BY video_id, start_ms, id").fetchall()
    scenes = []
    for vid, path, name in vids:
        mine = [r for r in rows if r[1] == vid]
        ts = [(r[3] + r[4]) // 2 for r in mine]
        got = dict(fr.extract_frames_at(path, ts))
        for r, t in zip(mine, ts):
            img = got.get(t)
            if img is None:
                continue
            key = f"v{vid}_{r[3]:07d}_{r[4]:07d}"
            img.save(work / "frames" / f"{key}.png")
            scenes.append({"key": key, "video": name, "video_id": vid, "start_ms": r[3], "end_ms": r[4],
                           "median_ms": t, "parent": r[5], "db_id": r[0]})
        print(f"{name}: {len(mine)} scenes", flush=True)
    (work / "manifest.json").write_text(json.dumps({"db": args.db, "scenes": scenes}, indent=1))
    print(f"{len(scenes)} frames -> {work}")


# ── embeddings ──────────────────────────────────────────────────────────────
def load_queries() -> list[dict]:
    return json.loads((HERE / "search_queries.json").read_text())["queries"]


class _Embedder:
    """Uniform image/text embedding over the candidate families."""

    def __init__(self, model: str, backend: str, patches: int):
        import torch
        self.model_id, self.backend, self.patches = model, backend, patches
        self.device = torch.device(backend)
        self.dtype = torch.float16 if backend in ("cuda", "mps") else torch.float32
        if model.startswith(("hf-hub:", "open_clip:")):
            import open_clip
            name = model.split(":", 1)[1] if model.startswith("open_clip:") else model
            m, _, pre = open_clip.create_model_and_transforms(name)
            self.kind = "open_clip"
            self.net = m.eval().to(self.device, self.dtype)
            self.pre = pre
            self.tok = open_clip.get_tokenizer(name)
        elif "naflex" in model:
            from backend.models.device import DeviceInfo
            from backend.models.siglip2 import SigLIP2Model
            self.kind = "app_naflex"
            dev = DeviceInfo(backend, backend, self.device, self.dtype)
            self.app = SigLIP2Model(dev, checkpoint=model, idle_offload_seconds=0, max_num_patches=patches)
            self.app._ensure_loaded()
        else:
            from transformers import AutoModel, AutoProcessor
            self.kind = "hf_fixed"
            self.proc = AutoProcessor.from_pretrained(model)
            self.net = AutoModel.from_pretrained(model, torch_dtype=self.dtype,
                                                 attn_implementation="sdpa").eval().to(self.device)

    def images(self, imgs) -> np.ndarray:
        import torch
        if self.kind == "app_naflex":
            return self.app.embed_images(imgs)
        with torch.inference_mode():
            if self.kind == "open_clip":
                x = torch.stack([self.pre(im) for im in imgs]).to(self.device, self.dtype)
                f = self.net.encode_image(x)
            else:
                x = self.proc(images=imgs, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
                f = self.net.get_image_features(pixel_values=x)
                f = f.pooler_output if hasattr(f, "pooler_output") else f
            f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return f.float().cpu().numpy()

    def text(self, q: str) -> np.ndarray:
        import torch
        if self.kind == "app_naflex":
            return self.app.embed_text(q)
        with torch.inference_mode():
            if self.kind == "open_clip":
                t = self.tok([q]).to(self.device)
                f = self.net.encode_text(t)
            else:
                t = self.proc(text=[q.lower()], padding="max_length", max_length=64, return_tensors="pt")
                t = {k: v.to(self.device) for k, v in t.items()}
                f = self.net.get_text_features(**t)
                f = f.pooler_output if hasattr(f, "pooler_output") else f
            f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return f.float().cpu().numpy()[0]


def _slug(model: str, patches: int | None = None) -> str:
    s = model.replace("/", "__").replace(":", "_")
    if patches and "naflex" in model:
        s += f"_p{patches}"
    return s


def cmd_embed(args) -> None:
    import torch
    backend = args.backend or _backend()
    _cap_memory(args.vram_gb, backend)
    work = Path(args.work)
    man, imgs = _load_frames(work)
    if args.limit:
        imgs = imgs[:args.limit]
    res = {"model": args.model, "patches": args.patches, "backend": backend, "batch": args.batch,
           "vram_cap_gb": args.vram_gb, "n": len(imgs), "sim_gb": os.environ.get("AMV_VRAM_LIMIT_GB"),
           "mps_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"), "suffix": args.suffix,
           "coreml": os.environ.get("AMV_COREML", "1") != "0"}
    sampler = _MemSampler(backend).start()
    t0 = time.perf_counter()
    try:
        emb = _Embedder(args.model, backend, args.patches)
        first = emb.images(imgs[:args.batch])
        _sync(backend)
        res["load_s"] = round(time.perf_counter() - t0, 2)
        times = []
        out = [first]
        for rep in range(args.repeat):
            t1 = time.perf_counter()
            chunks = [emb.images(imgs[i:i + args.batch]) for i in range(args.batch, len(imgs), args.batch)]
            _sync(backend)
            times.append(time.perf_counter() - t1)
            if rep == 0:
                out += chunks
        n_timed = len(imgs) - args.batch
        res["ips_runs"] = [round(n_timed / t, 1) for t in times]
        res["ips"] = round(n_timed / float(np.median(times)), 1)
        E = np.concatenate(out).astype(np.float32)
        qs = load_queries()
        t2 = time.perf_counter()
        T = np.stack([emb.text(q["text"]) for q in qs]).astype(np.float32)
        _sync(backend)
        res["text_ms"] = round((time.perf_counter() - t2) * 1000 / len(qs), 1)
        res["ok"] = True
        if not args.no_save:
            np.savez(work / f"emb_{_slug(args.model, args.patches)}{args.suffix}.npz", images=E, texts=T,
                     keys=np.array([s["key"] for s in man["scenes"][:len(imgs)]]))
    except Exception as exc:  # noqa: BLE001 - recorded, the matrix needs the failure
        res["ok"] = False
        res["oom"] = _is_oom(exc)
        res["error"] = str(exc)[:300]
    sampler.stop()
    res.update(_mem_report(backend, sampler))
    print(json.dumps(res), flush=True)
    with open(work / "embed_runs.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


# ── tagger ──────────────────────────────────────────────────────────────────
def _make_tagger(model: str, backend: str):
    if model == "pixai-v1.0":
        from tests.eval.pixai import PixAIV1Tagger
        return PixAIV1Tagger(model, backend)
    if model.startswith("pixai"):
        from tests.eval.pixai import PixAITagger
        return PixAITagger(model, backend)
    from backend.models.wd_tagger import WDTaggerModel
    t = WDTaggerModel(repo_id=model, general_threshold=0.35, character_threshold=0.85,
                      device_backend=backend)
    t._ensure_loaded()
    cml = getattr(t._session, "_cml", None)
    if cml is not None:
        # Core ML worker (AMV_COREML=0 measures plain MPS): time it only
        # once compiled, not the MPS fallback that serves meanwhile.
        t0 = time.monotonic()
        while not cml.usable() and not cml.failed and time.monotonic() - t0 < 900:
            time.sleep(0.5)
        print(f"coreml worker ready={cml.usable()} after {time.monotonic() - t0:.1f}s failed={cml.failed}",
              file=sys.stderr, flush=True)
    return t


def cmd_tag(args) -> None:
    backend = args.backend or _backend()
    _cap_memory(args.vram_gb, backend)
    work = Path(args.work)
    man, imgs = _load_frames(work)
    if args.limit:
        imgs = imgs[:args.limit]
    res = {"model": args.model, "backend": backend, "batch": args.batch, "vram_cap_gb": args.vram_gb,
           "n": len(imgs), "sim_gb": os.environ.get("AMV_VRAM_LIMIT_GB"),
           "mps_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"), "suffix": args.suffix,
           "coreml": os.environ.get("AMV_COREML", "1") != "0"}
    sampler = _MemSampler(backend).start()
    t0 = time.perf_counter()
    try:
        tg = _make_tagger(args.model, backend)
        first = tg._run_batch(imgs[:args.batch])
        _sync(backend)
        res["load_s"] = round(time.perf_counter() - t0, 2)
        res["provider"] = tg.active_providers() if hasattr(tg, "active_providers") else None
        times, out = [], [np.asarray(first)]
        for rep in range(args.repeat):
            t1 = time.perf_counter()
            chunks = [np.asarray(tg._run_batch(imgs[i:i + args.batch]))
                      for i in range(args.batch, len(imgs), args.batch)]
            _sync(backend)
            times.append(time.perf_counter() - t1)
            if rep == 0:
                out += chunks
        n_timed = len(imgs) - args.batch
        res["ips_runs"] = [round(n_timed / t, 1) for t in times]
        res["ips"] = round(n_timed / float(np.median(times)), 1)
        S = np.concatenate(out).astype(np.float16)
        labels = [(n, c) for n, c in tg._labels]
        res["n_labels"] = len(labels)
        res["ok"] = True
        if not args.no_save:
            np.savez_compressed(work / f"tags_{_slug(args.model)}{args.suffix}.npz", scores=S,
                                names=np.array([n for n, _ in labels]),
                                cats=np.array([c for _, c in labels], dtype=np.int16),
                                keys=np.array([s["key"] for s in man["scenes"][:len(imgs)]]))
    except Exception as exc:  # noqa: BLE001
        res["ok"] = False
        res["oom"] = _is_oom(exc)
        res["error"] = str(exc)[:300]
    sampler.stop()
    res.update(_mem_report(backend, sampler))
    print(json.dumps(res), flush=True)
    with open(work / "tag_runs.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


# ── search scoring ──────────────────────────────────────────────────────────
def _emb_files(work: Path) -> dict[str, Path]:
    return {p.stem[4:]: p for p in sorted(work.glob("emb_*.npz"))}


def _ranked(work: Path, top: int = 10, similar_seeds: list[str] | None = None) -> dict:
    """{model: {"q:<id>": [keys], "s:<seed>": [keys]}} — child scenes and the
    seed itself are excluded from 'similar'."""
    qs = load_queries()
    out = {}
    for name, p in _emb_files(work).items():
        z = np.load(p)
        E, T, keys = z["images"], z["texts"], list(z["keys"])
        r = {}
        for q, t in zip(qs, T):
            sims = E @ t
            r[f"q:{q['id']}"] = [keys[i] for i in np.argsort(-sims)[:top]]
        for seed in similar_seeds or []:
            if seed not in keys:
                continue
            i0 = keys.index(seed)
            sims = E @ E[i0]
            # same scene or a sub-segment of the same shot is not "similar"
            v0, s0, e0 = _parse_key(seed)
            order = [i for i in np.argsort(-sims) if not _overlaps(keys[i], v0, s0, e0)]
            r[f"s:{seed}"] = [keys[i] for i in order[:5]]
        out[name] = r
    return out


def _parse_key(k: str):
    v, s, e = k.split("_")
    return v, int(s), int(e)


def _overlaps(k: str, v0, s0, e0) -> bool:
    v, s, e = _parse_key(k)
    return v == v0 and s < e0 + 2000 and e > s0 - 2000


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def cmd_score_search(args) -> None:
    work = Path(args.work)
    j = json.loads((HERE / "search_judgments.json").read_text())
    judg = j["judgments"]
    seeds = j.get("similar_seeds", [])
    ranked = _ranked(work, similar_seeds=seeds)
    rows = {}
    for model, r in ranked.items():
        p10, ndcg, sim5, unjudged = [], [], [], 0
        per_q = {}
        for qk, keys in r.items():
            grades = judg.get(qk, {})
            gains = []
            for k in keys:
                g = grades.get(k)
                if g is None:
                    unjudged += 1
                    g = 0
                gains.append(float(g))
            if qk.startswith("q:"):
                ideal = sorted([float(v) for v in grades.values()], reverse=True)[:10]
                n = _dcg(gains) / _dcg(ideal) if ideal and _dcg(ideal) > 0 else 0.0
                p = sum(1 for g in gains if g >= 1) / 10
                p10.append(p)
                ndcg.append(n)
                per_q[qk] = {"p10": p, "ndcg10": round(n, 3)}
            else:
                s = sum(1 for g in gains if g >= 1) / 5
                sim5.append(s)
                per_q[qk] = {"p5": s}
        rows[model] = {"p10": round(float(np.mean(p10)), 3) if p10 else None,
                       "ndcg10": round(float(np.mean(ndcg)), 3) if ndcg else None,
                       "similar_p5": round(float(np.mean(sim5)), 3) if sim5 else None,
                       "unjudged": unjudged, "per_query": per_q}
    print(json.dumps({m: {k: v for k, v in r.items() if k != "per_query"} for m, r in rows.items()}, indent=1))
    (work / "search_scores.json").write_text(json.dumps(rows, indent=1))


def cmd_pool(args) -> None:
    """Union of the top-10 (top-5 similar) of every model, minus what is
    already judged: what still needs a human look."""
    work = Path(args.work)
    jp = HERE / "search_judgments.json"
    j = json.loads(jp.read_text()) if jp.exists() else {"judgments": {}, "similar_seeds": []}
    ranked = _ranked(work, similar_seeds=j.get("similar_seeds", []))
    pool: dict[str, list[str]] = {}
    for r in ranked.values():
        for qk, keys in r.items():
            done = j["judgments"].get(qk, {})
            for k in keys:
                if k not in done and k not in pool.setdefault(qk, []):
                    pool[qk].append(k)
    pool = {k: v for k, v in pool.items() if v}
    (work / "pool.json").write_text(json.dumps(pool, indent=1))
    print(sum(len(v) for v in pool.values()), "unjudged pairs in", len(pool), "queries")


def cmd_sheets(args) -> None:
    """One labelled contact sheet per query of the pending pool."""
    from PIL import Image, ImageDraw
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = json.loads((work / "pool.json").read_text())
    qs = {f"q:{q['id']}": q["text"] for q in load_queries()}
    cols, tw, th = 6, 256, 144
    for qk, keys in pool.items():
        extra = []
        if qk.startswith("s:"):
            extra = [qk[2:]]
        items = extra + keys
        rows = math.ceil(len(items) / cols)
        sheet = Image.new("RGB", (cols * tw, rows * (th + 18) + 24), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        d.text((4, 4), f"{qk}  {qs.get(qk, '(similar to #0)')}", fill=(255, 255, 0))
        for i, k in enumerate(items):
            im = Image.open(work / "frames" / f"{k}.png").convert("RGB").resize((tw, th))
            x, y = (i % cols) * tw, 24 + (i // cols) * (th + 18)
            sheet.paste(im, (x, y))
            d.text((x + 3, y + th + 2), f"#{i} {k}" if not extra or i else f"SEED {k}",
                   fill=(255, 255, 255) if not extra or i else (0, 255, 0))
        safe = qk.replace(":", "_").replace("/", "_")
        sheet.save(out / f"{safe}.jpg", quality=85)
    print(len(pool), "sheets in", out)


# ── tagger scoring ──────────────────────────────────────────────────────────
def cmd_score_tags(args) -> None:
    work = Path(args.work)
    truth = json.loads((HERE / "tagger_truth.json").read_text())
    ann = truth["scenes"]
    tag_map = truth["tag_map"]            # our visual tag -> danbooru names (any of)
    char_map = truth["character_map"]     # character -> danbooru names (any of)
    out = {}
    for p in sorted(work.glob("tags_*.npz")):
        z = np.load(p)
        S, names, cats, keys = z["scores"].astype(np.float32), list(z["names"]), z["cats"], list(z["keys"])
        idx = {n: i for i, n in enumerate(names)}
        model = p.stem[5:]
        gthr = truth.get("general_threshold", 0.35)
        cthr = truth.get("character_threshold", 0.85)
        res = {}
        for label, mapping, thr in (("characters", char_map, cthr), ("tags", tag_map, gthr)):
            tp = fp = fn = 0
            per = {}
            for name, aliases in mapping.items():
                cols = [idx[a] for a in aliases if a in idx]
                ctp = cfp = cfn = 0
                for k, a in ann.items():
                    if k not in keys:
                        continue
                    present = name in a.get(label, [])
                    score = max((S[keys.index(k), c] for c in cols), default=0.0)
                    hit = bool(score >= thr)
                    ctp += hit and present
                    cfp += hit and not present
                    cfn += (not hit) and present
                per[name] = {"tp": int(ctp), "fp": int(cfp), "fn": int(cfn), "known": bool(cols)}
                tp, fp, fn = tp + ctp, fp + cfp, fn + cfn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            res[label] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "per": per}
        res["recognised_characters"] = sorted(n for n, v in res["characters"]["per"].items() if v["tp"] > 0)
        out[model] = res
    print(json.dumps({m: {k: (v if k == "recognised_characters" else {kk: vv for kk, vv in v.items() if kk != "per"})
                          for k, v in r.items()} for m, r in out.items()}, indent=1))
    (work / "tag_scores.json").write_text(json.dumps(out, indent=1))


def cmd_tag_sheet(args) -> None:
    """Contact sheets of the annotation scenes, with each model's character
    tags above the threshold (for the hand annotation)."""
    from PIL import Image, ImageDraw
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = json.loads(Path(args.keys).read_text())
    cols, tw, th = 4, 384, 216
    for start in range(0, len(keys), 12):
        part = keys[start:start + 12]
        rows = math.ceil(len(part) / cols)
        sheet = Image.new("RGB", (cols * tw, rows * (th + 16)), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for i, k in enumerate(part):
            im = Image.open(work / "frames" / f"{k}.png").convert("RGB").resize((tw, th))
            x, y = (i % cols) * tw, (i // cols) * (th + 16)
            sheet.paste(im, (x, y))
            d.text((x + 3, y + th + 1), f"#{start + i} {k}", fill=(255, 255, 0))
        sheet.save(out / f"tags_{start:03d}.jpg", quality=88)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--db", required=True)
    d.add_argument("--work", required=True)
    for name in ("embed", "tag"):
        e = sub.add_parser(name)
        e.add_argument("--work", required=True)
        e.add_argument("--model", required=True)
        e.add_argument("--patches", type=int, default=128)
        e.add_argument("--batch", type=int, default=64)
        e.add_argument("--repeat", type=int, default=1)
        e.add_argument("--vram-gb", type=float)
        e.add_argument("--backend")
        e.add_argument("--limit", type=int)
        e.add_argument("--suffix", default="")
        e.add_argument("--no-save", action="store_true")
    for name in ("score-search", "pool", "score-tags"):
        s = sub.add_parser(name)
        s.add_argument("--work", required=True)
    s = sub.add_parser("sheets")
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("tag-sheet")
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--keys", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    {"dump": cmd_dump, "embed": cmd_embed, "tag": cmd_tag, "score-search": cmd_score_search,
     "pool": cmd_pool, "sheets": cmd_sheets, "score-tags": cmd_score_tags,
     "tag-sheet": cmd_tag_sheet}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
