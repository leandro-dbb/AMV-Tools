# AMV Tools on NVIDIA GPUs — performance notes (0.4)

Everything below was measured on real hardware with real anime, not
synthetic clips. The benchmark is `tests/bench_cuda.py`; every number in the
tables is the median of two runs (same machine, same files, page cache warm).

## Test machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB, driver 580.196.07 (CUDA 13.0), PCIe 4.0 x16 |
| CPU / RAM | 6 vCPU of an AMD Ryzen 9 5900X (VM), 15 GB RAM |
| OS | Debian 12, Python 3.13, torch 2.11.0+cu126, onnxruntime-gpu 1.26, transformers 5.7 |
| ffmpeg | before: bundled imageio-ffmpeg 7.0.2 (static, **no NVDEC/NVENC**); after: Debian 5.1.9 (NVDEC, `scale_cuda`, `overlay_cuda`, NVENC) |
| Footage | *Dragon Ball Super* 01 and 02 (1080p H.264 23.976, ~23 min each), a 4-min extract of episode 01, and a 1080p HEVC 10-bit episode for cross-checks |

## Method

```bash
# indexing (end to end + isolated stages), search, export — 4-min extract / full episode
uv run python tests/bench_cuda.py --video clip.mkv --sections index,search,export --label after --out after.json
# finishing on a 5 s 1080p window (upscale x2/x4, RIFE slow-mo x2 / 60 fps, depth, cutout, SAM 2)
uv run python tests/bench_cuda.py --video clip.mkv --sections finish --finish-start-ms 90000
# a queue of episodes, as a user batch runs it
uv run python tests/bench_cuda.py --video "ep01.mkv|ep02.mkv" --sections queue
# before/after table, median of the runs on each side
uv run python tests/bench_cuda.py --compare before-r1.json,before-r2.json after-r1.json,after-r2.json
# 8 GB card simulation (caps this process' CUDA allocator)
uv run python tests/bench_cuda.py --video clip.mkv --sections index,finish --vram-fraction 0.333
```

Per stage the bench records wall time, throughput, mean GPU / NVDEC / NVENC
utilisation and max VRAM (`nvidia-smi` every 200 ms), the torch allocator peak
and the peak RSS of the whole process tree (ffmpeg children included). The
"before" runs use the code of `be96c09` with the same bench file.

Quality guard rails (`--sections quality --save-ref` then `--compare-ref`, on
48 frames spread over the extract and 6 consecutive 1080p frames):

| check | result |
|---|---|
| SigLIP 2 embeddings, GPU preprocessing + fp16 vs CPU preprocessing + fp32 | cosine min **0.9988**, mean 0.9996 |
| wd-tagger, PyTorch fp16 vs ONNX Runtime fp32 | top-5 tags identical **48/48**; stored tag sets (≥ 0.50) 47/48 |
| hard cuts, single NVDEC pass vs the two-pass detector | **identical** (79/79, 366/366, 419/419, 261/261 HEVC) |
| median frames, single pass vs the extractor at the same timestamp | ≥ 37.9 dB (median 40.1 dB) — same frame, same conversion |
| median frame timecodes vs a PyAV reference | exact on 8/8 targets (the old extractor was one frame late on 2/8) |
| upscale x2 (GPU area downscale + channels_last) vs before | 71 dB |
| RIFE, depth | identical |
| cutout matte (BiRefNet GPU resize) vs CPU transform | 44 dB (alpha RMS 0.006) |
| temporal stabilisation GPU vs CPU | mean abs diff 1e-6, 66 dB |
| ProRes 4444 cutout, raw pipes vs PNG round-trip | **bit-identical** |
| I420 conversion in the writer (OpenCV) vs ffmpeg swscale | luma within 1/255 (68 dB), chroma 52 dB |
| NVENC "match source" export vs x264 CRF 18 medium | 50.8 dB vs 45.2 dB luma PSNR (vs a lossless reference) |

## Results — full pipeline (chantier 2)

`tests/bench_pipeline.py` replaces `tests/bench_cuda.py`: it drives the real
sidecar like the UI (3-episode index queue with Search+ and hover proxies,
18 searches, 10 clip exports + 4 timelines, upscale x2 / RIFE x2 / depth on a
5 s clip). One run per column, page cache warm, nothing else running on the
Proxmox host.

| | v0.3 (estimated ¹) | before chantier 2 (`8b12280`) | after (`9c8e971`) | round 3 (`638789f`) ³ |
|---|---:|---:|---:|---:|
| **full pipeline** | ~45 min 30 | 8 min 12 | 5 min 49 (x1.41) | **4 min 32 - 4 min 43** |
| season: 3 episodes indexed + hover proxies | ~39 min 50 | 336.6 s | 269.1 s | **193.7 - 203.9 s** |
| — decode + cuts + Search+ motion | | 177.7 s (GPU 5 %) | 189.9 s ² (GPU 17 %) | 163.6 - 169.8 s (NVDEC 61-65 %) |
| — tags / embeddings | | 23.3 / 16.3 s | 22.5 / 23.8 s | **10.7 / 6.2 s** |
| — hover proxies | | 115.6 s | 28.2 s | **7.3 s** |
| 18 searches (first one cold) | ~45 s | 11.8 s | 9.1 s | 7.6 - 7.7 s |
| 10 clip exports + 4 timelines | ~59 s | 29.6 s | 12.4 s | 11.5 - 12.9 s |
| upscale x2 + RIFE x2 + depth (5 s 1080p) | 237.6 s | 114.0 s | 58.8 s | 57.9 - 60.3 s |

¹ v0.3 (`be96c09`) was not re-run: episode 01 numbers of round 1 x 3, and
per-clip costs. ² NVDEC-bound (one decoder on the 3090) and noisy on the
shared VM host: 149 s in an otherwise identical run. ³ Two runs interleaved
with two runs of the round-2 code (`a42f217`) on the same host: 329.9 / 382.9 s
for round 2 against 272.0 / 283.4 s for round 3; the CPU of the 6-vCPU VM,
shared with other VMs, is the limit of the season (round 3 report).

Per-stage tables, GPU utilisation, what moved from the CPU to the GPU, what
stays on the CPU and why, quality checks and rejected ideas:
[docs/reports/chantier-2-pipeline-nvidia.md](reports/chantier-2-pipeline-nvidia.md)
(in French).
