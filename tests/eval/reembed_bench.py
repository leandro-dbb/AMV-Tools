"""Time the embeddings-only re-index on a real indexed library.

    python tests/eval/reembed_bench.py DATA_DIR google/siglip2-so400m-patch16-naflex

DATA_DIR is a bench data folder (tests/bench_pipeline.py --data-dir) indexed
with another embedding model; it is used in place (copy it first).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests"))

from bench_pipeline import Sidecar  # noqa: E402


def main() -> int:
    data, target = Path(sys.argv[1]), sys.argv[2]
    sc = Sidecar(_REPO, sys.executable, data, data / "reembed.log")
    sc.start()
    try:
        before = sc.get("/api/models/tiers")["embedding"]
        q0 = sc.post("/api/search", {"query": "beerus", "top_k": 10, "threshold": 0.0})
        t0 = time.time()
        r = sc.post("/api/models/reembed", {"model": target})
        job = r.get("job") or {}
        while True:
            job = sc.get("/api/models/reembed").get("job") or {}
            if job.get("done") or not job.get("running", True):
                break
            time.sleep(0.5)
        dt = time.time() - t0
        after = sc.get("/api/models/tiers")["embedding"]
        q1 = sc.post("/api/search", {"query": "beerus", "top_k": 10, "threshold": 0.0})
        out = {"target": target, "seconds": round(dt, 1), "job": job, "before": before, "after": after,
               "results_before": len(q0.get("results", [])), "results_after": len(q1.get("results", []))}
        t1 = time.time()
        sc.post("/api/models/reembed", {"model": before["db_model"]})
        while (sc.get("/api/models/reembed").get("job") or {}).get("running"):
            time.sleep(0.2)
        out["switch_back_s"] = round(time.time() - t1, 2)
        print(json.dumps(out, indent=1))
    finally:
        sc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
