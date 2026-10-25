"""Occupy the GPU so that only N GB stay free: a smaller card, for real.

    python tests/eval/ballast.py 8 &                  # leave 8 GB free
    AMV_VRAM_LIMIT_GB=8 python tests/bench_pipeline.py ...

Everything started afterwards (PyTorch, ONNX Runtime, ffmpeg's NVDEC/NVENC)
sees only N GB of free VRAM; AMV_VRAM_LIMIT_GB makes the app pick the batch
sizes / tier of an N GB card. Kill the process to give the memory back.
"""
import sys
import time

import torch

target = float(sys.argv[1])
free, _total = torch.cuda.mem_get_info()
take = free - target * 2**30
x = torch.empty(max(0, int(take)), dtype=torch.uint8, device="cuda") if take > 0 else None
print(f"ballast: took {take / 2**30:.2f} GB, free now {torch.cuda.mem_get_info()[0] / 2**30:.2f} GB", flush=True)
time.sleep(10**9)
