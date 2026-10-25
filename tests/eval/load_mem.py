"""Peak memory while loading a SigLIP 2 checkpoint, two ways (Apple Silicon).

    python tests/eval/load_mem.py google/siglip2-so400m-patch16-naflex [--device-map]

Prints the peak physical footprint of the process during the load (macOS
``proc_pid_rusage``; RSS elsewhere) and the load time. ``--device-map``
asks transformers to create the weights directly on the accelerator.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from bench_pipeline import _footprint  # noqa: E402
    import resource

    import torch
    from transformers import Siglip2Model

    repo = sys.argv[1]
    dm = "--device-map" in sys.argv
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    peak = [0]
    stop = threading.Event()

    def anon() -> int:
        # Linux: anonymous RSS (mapped weight files excluded), else footprint
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("RssAnon:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return _footprint(os.getpid()) or 0

    def sample():
        while not stop.wait(0.02):
            peak[0] = max(peak[0], anon())
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    t0 = time.perf_counter()
    if dm:
        m = Siglip2Model.from_pretrained(repo, dtype=torch.float16, device_map=dev)
    else:
        m = Siglip2Model.from_pretrained(repo, dtype=torch.float16).to(dev)
    if dev == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    time.sleep(0.2)
    stop.set()
    t.join()
    print(json.dumps({"repo": repo, "device_map": dm, "device": dev, "load_s": round(dt, 2),
                      "peak_anon_or_footprint_gb": round(peak[0] / 2**30, 2) if peak[0] else None,
                      "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                         / (2**30 if sys.platform == "darwin" else 2**20), 2),
                      "param_device": str(next(m.parameters()).device)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
