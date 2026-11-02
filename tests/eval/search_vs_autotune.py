"""How long the first search waits when the background autotune is running.

    python -m tests.eval.search_vs_autotune --data-dir DIR [--repo TREE] [--after 25] [--autotune 0|1]

``DIR``: a data folder with an indexed library (a ``bench_pipeline`` run's,
copied: it is modified). A fresh sidecar starts on it with no autotune
result (``autotune.json`` removed, so the bench runs ``START_DELAY_S``
after start-up); ``--after`` seconds later — mid-bench — a search is sent
and timed, then a second one. ``--autotune 0``: the same without the bench
(the reference: SigLIP cold start only).

Measured for docs/reports/windows-rtx3090.md (point 5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    from tests.bench_pipeline import Sidecar
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--after", type=float, default=25.0)
    p.add_argument("--at-step", help="search once the bench is at this step instead (e.g. batch_size)")
    p.add_argument("--autotune", default="1")
    p.add_argument("--query", default="Beerus endormi")
    p.add_argument("--json")
    a = p.parse_args()
    data = Path(a.data_dir)
    (data / "autotune.json").unlink(missing_ok=True)
    (data / ".setup_complete").write_text("bench\n")
    os.environ["AMV_AUTOTUNE"] = a.autotune
    sc = Sidecar(Path(a.repo), a.python, data, data / "search-vs-autotune.log")
    t_start = sc.start()
    out = {"autotune": a.autotune, "sidecar_ready_s": round(t_start, 1), "after_s": a.after}
    try:
        time.sleep(max(0.0, a.after - t_start))
        if a.at_step:
            deadline = time.time() + 600
            while time.time() < deadline and sc.get("/api/autotune").get("step") != a.at_step:
                time.sleep(0.2)
            time.sleep(1.0)
        st0 = sc.get("/api/autotune")
        out["autotune_state_before"] = st0.get("state")
        out["autotune_step_before"] = st0.get("step")
        t_search = time.time()
        lat = []
        for q in (a.query, "gros plan visage"):
            t0 = time.perf_counter()
            res = sc.post("/api/search", {"query": q, "top_k": 50, "threshold": 0.0})
            lat.append(round(time.perf_counter() - t0, 2))
            n = len(res.get("results", res) if isinstance(res, dict) else res)
            out.setdefault("results", []).append(n)
        out["latency_s"] = lat
        out["autotune_state_after"] = sc.get("/api/autotune").get("state")
        # does the bench finish once the app is quiet?
        deadline = time.time() + 300
        while time.time() < deadline:
            st = sc.get("/api/autotune")
            if st.get("state") in ("done", "failed", "disabled"):
                break
            time.sleep(2)
        out["autotune_final"] = st.get("state")
        out["autotune_done_after_search_s"] = round(time.time() - t_search, 1)
        out["autotune_entry_settings"] = sorted(((st.get("entry") or {}).get("settings") or {}).keys())
    finally:
        sc.stop()
    print(json.dumps(out), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
