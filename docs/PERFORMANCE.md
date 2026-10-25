# Performance

Measured numbers for AMV Tools on real footage, the method behind them, and
how to reproduce them on another machine. Everything here comes from
`tests/bench_platform.py`; no number is estimated.

## Apple Silicon — MacBook Pro M1 Pro (8P+2E), 16 GB, macOS 26.6

Footage: a full 24-min Dragon Ball Super episode (1080p H.264, 23.976 fps,
33 512 frames) and a 4-min extract of it. Default settings of a fresh
install, i.e. the onboarding's recommended **Accurate** profile (balanced
mode + sub-segmentation); the **Fast** profile (no sub-segmentation) is
reported separately.

**Before** is the 0.4 branch point (`be96c09`) — but that build cannot index
on Apple Silicon at all (see *Stability* below): the wd-tagger aborts the
process on its first batch, and with the tagger on CPU SigLIP 2 fails or
stores corrupted embeddings. *Before* is therefore that code with the two
smallest workarounds that let it finish: the wd-tagger forced onto the ONNX
CPU provider (`--tagger-cpu`) and blocking SigLIP host-to-device copies. It
is, if anything, flattering to the old build. **After** is this branch.

All values: median of 2 interleaved runs (before, after, before, after).

**Indexing — Accurate profile (default)**

| | Before | After | |
|---|---:|---:|---:|
| 4-min extract, total | 198.8 s | **28.5 s** | ×7.0 |
| 24-min episode, total | 1 165.5 s (19 min 25) | **121.1 s** (2 min 01) | ×9.6 |
| — cut detection + decode | 169.3 s | 112.8 s ¹ | |
| — extra decodes (sub-segmentation samples, medians) | 328.8 s | 0 | |
| — wd-tagger (incl. sub-segmentation samples) | 237.9 s (2.5 img/s, CPU EP ²) | 38.3 s (15.8 img/s, MPS) | |
| — SigLIP 2 | 6.7 s | 7.3 s | |
| — Search+ motion analysis (CPU, overlapped) | 32.7 s | 13.7 s | |
| segments (incl. sub-segments) | 592 | 605 | |
| peak memory (process footprint) | 2.8 GB | 4.9 GB | |
| swap-outs during the run | 1.2 k pages | 1.4 k pages ³ | |

**Indexing — Fast profile (no sub-segmentation)**

| | Before | After | |
|---|---:|---:|---:|
| 4-min extract, total | 94.9 s | **14.9 s** | ×6.4 |
| 24-min episode, total | 531.8 s (8 min 52) | **43.3 s** | ×12.3 |
| — cut detection + decode | 169.7 s | 39.4 s ¹ | |
| — median-frame extraction (2nd decode) | 188.5 s | 0 | |
| — wd-tagger | 160.9 s (2.5 img/s) | 27.2 s, overlapped with decode | |
| — SigLIP 2 | 4.6 s | 6.1 s, overlapped | |
| peak memory / swap-outs | 2.6 GB / 1.6 k pages | 4.5 GB / 0 | |

¹ After: the decode also waits for the GPU (tagging/embedding run during the
decode), so this column is the critical path, not pure decoding (pure
software decode of the episode: ~37 s, ~900 fps).
² The original Core ML tagger crashes; see above.
³ Both runs show ~20 MB of system swap-outs from background macOS services;
no run of either build swapped more than that during indexing.

**Search, export, finishing, Mask Mode**

| | Before | After | |
|---|---:|---:|---:|
| First text search (loads SigLIP 2) | 5.32 s | 5.17 s | |
| Text search / image search / similar shots (418 scenes) | 16 / 30 / 3.1 ms | 16 / 28 / 2.8 ms | |
| Export 10 s 1080p, libx264 CRF 18 | 2.57 s — 5.6 Mb/s, VMAF 96.8 | 2.50 s — same | |
| Export 10 s 1080p, h264_videotoolbox (adaptive preset) | 1.83 s — 11.8 Mb/s, VMAF 97.2 | 1.86 s — same | |
| Hover proxy (3 s) | 0.71 s | **0.39 s** | −45 % |
| Upscale x2 animevideov3, 1080p | 0.876 s/frame | **0.542 s/frame** | −38 % |
| — 120-frame job (incl. 4K libx264 encode) | 110.0 s, 6.4 GB, swapped | **72.0 s, 3.6 GB, no swap** | |
| RIFE 4.25 in-between, 1080p | 0.201 s/frame | 0.201 s/frame | |
| — 120-frame jobs, x2 slow-mo / 60 fps | 26.1 s / 58.9 s | 26.2 s / 58.9 s | |
| Depth Anything V2 Small, 1080p | 0.179 s/frame | 0.179 s/frame | |
| Mask Mode Auto, ISNet anime (Core ML), 1080×608 | 0.350 s/frame | 0.340 s/frame | |
| Mask Mode Manual, SAM 2.1 tiny, 48 frames | 0.655 s/frame, **13.0 GB, 3.7 GB swapped** | **0.598 s/frame, 7.5 GB, no swap** | |

Output quality of *after* against *before* (bench `--ref`): upscale PSNR
56.4 dB (max 7/255), RIFE / depth / ISNet bit-identical. On identical frames
the MPS wd-tagger matches the ONNX CPU reference (stored tag set identical on
98 % of 200 episode frames, Jaccard 0.998, max score difference 0.005) and
SigLIP 2 fp16 on MPS matches fp32 on CPU (cosine ≥ 0.99945, none below
0.995). End to end the indexed embeddings differ more (median cosine 0.983 on
the 4-min extract) because the representative frame moved: cuts now land up
to 3 frames earlier, medians come from the nearest sampled frame (≤ 2
frames) and frames are scaled bicubic (closer to a Lanczos reference than
the previous VideoToolbox scaler). 88 of 92 cuts of the extract agree within
4 frames; camera-motion labels agree on 98 % of scenes.


### Stability (16 GB)

- **Crash fixed — wd-tagger on Core ML**: with the tagger's dynamic batch
  dimension, onnxruntime 1.26 split the ViT into 38 Core ML partitions and
  the GPU compile of that dynamic MLProgram hit an MPSGraph assertion that
  aborts the process (`original module failed verification`, SIGABRT,
  reproduced on every run). The sidecar died on the first tag batch.
- **Silent corruption fixed — SigLIP 2 on MPS**: `non_blocking=True` copies
  from pageable memory raced with the processor tensors being freed: 30/30
  batches reached the GPU corrupted (garbage embeddings, or `Height of
  resized positional embeddings must be positive`). Copies are blocking on
  MPS now.
- **No swap**: PyTorch's MPS allocator only recycles its cache past 1.4x
  Metal's working set (~16.5 GB here). SAM 2 grew a 12 GB cache over a
  ~2 GB live set and swapped ~4 GB for a 2-s Mask Mode track. The sidecar
  sets `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25`: 7.5 GB footprint, no swap,
  nothing slower. No pass of the *after* column swapped beyond ~20 MB of
  background noise (`swapouts` in the
  JSON); *before*, SAM 2 swapped 3.7 GB and the upscale job ~110 MB.
- Models are released between passes (finishing jobs evict the indexing and
  mask models; the indexing queue frees the tagger at the end).
- Batch size: on Apple Silicon it barely matters (tagger and SigLIP gain
  < 10 % from 8 to 64), so Settings recommends 8 / 16 / 32 for 8 / 16-24 /
  32 GB+ Macs; the default (8) is what was measured.

### What was changed, and what each change measured

Kept (M1 Pro, gains measured in isolation):

| Change | Measured |
|---|---|
| Software (frame-threaded) decode instead of ffmpeg's VideoToolbox hwaccel, except HEVC > 1440p | 4-min 1080p H.264: 37.6 s → 5.5 s; HEVC 10-bit 1080p 15.0 → 9.3 s; HEVC 4K: tie (6.0 vs 6.5 s, VT kept) |
| One decode for cuts + median frames (+ sub-segmentation samples), instead of 2–3 | each removed pass is a full decode (~37 s per episode in software, ~190 s with VideoToolbox) |
| Tag/embed on the GPU while the CPU keeps decoding (bounded queue) | episode, Fast: 70.6 → 43.3 s; Accurate: 222 → 121 s |
| wd-tagger on PyTorch MPS (same checkpoint via timm) instead of ONNX/Core ML | 17 img/s (= Core ML), load 1.5 s instead of 32 s (126 s first time), no GIL freeze, no GB-sized compiled cache |
| Bicubic scaler for model-input frames | SigLIP cos vs a Lanczos reference 0.9965 (VT scaler 0.9868) |
| Upscale: fp16 + whole frame + x4→x2 on the GPU | 0.876 → 0.541 s/frame, PSNR 56.4 dB vs before |
| Hover proxies decoded in software | 0.72 → 0.39 s per proxy; one-pass proxies of 4 min 35.6 → 8.0 s |
| Core ML compute units CPUAndGPU (ISNet, tagger fallback) | tagger 17.6 vs 13.1 (ALL) vs 6.5 (CPU+ANE) img/s; ISNet 2.8 vs 2.6 vs 1.6 |
| MPS allocator low watermark 0.25 | SAM 2 footprint 13.1 → 7.5 GB, no swap |

Rejected (no net gain, or quality loss):

| Tried | Result |
|---|---|
| fp16 for RIFE | 0.196 → 0.199 s (M1's GPU runs fp16 at the fp32 rate) |
| fp16 for Depth Anything V2 | PSNR 10.6 dB vs fp32 — broken output |
| fp16 / batch 16-64 for SigLIP 2 | already fp16; +8 % at batch 64, not worth the memory |
| channels_last (upscale) | 0.766 → 0.791 s, slower |
| Real-ESRGAN through Core ML | 43 ms/tile (CPU+GPU), 140 ms (Neural Engine) vs 50 ms MPS |
| wd-tagger on Core ML with a pinned batch | works (crash fixed, kept as fallback) but 32 s session build holding the GIL |
| Core ML NeuralNetwork format | tagger 0.2 img/s |
| `empty_cache()` every N frames in SAM 2 | 11.0 → 8.3 GB, the watermark does better |
| Library-mean centering of character embeddings | AP 0.708 → 0.732, not the actual problem (threshold) |

No PyTorch op of SigLIP 2, the wd-tagger, Real-ESRGAN, RIFE, Depth Anything
or SAM 2.1 fell back to the CPU under `PYTORCH_ENABLE_MPS_FALLBACK` (the
bench records the fallback warnings; it was checked to catch e.g.
`aten::linalg_eig`).

### What is still slow, and why

- **Decoding** (~37–39 s per 24-min episode) is now the critical path of the
  Fast profile: libavcodec's H.264 decoder at ~900 fps on 10 cores is the
  ceiling; VideoToolbox would be slower through ffmpeg.
- **The wd-tagger** (ViT-B/16 at 448 px, ~17 img/s) bounds the Accurate
  profile: sub-segmentation embeds a frame every second of every scene
  longer than 4 s (~900 extra images per episode). The M1 GPU runs fp16 no
  faster than fp32, the Neural Engine was slower, Core ML equal.
- **Finishing** passes are GPU compute-bound (upscale ≈ 2.4 TFLOP per 1080p
  frame); Mask Mode SAM 2 still peaks at ~7.5 GB.
- The first text search of a session loads SigLIP 2 (~5 s).

### Round 2 — the full pipeline (`tests/bench_pipeline.py`)

One run of the whole app per column (3-episode season indexed as one
queue with hover proxies, 18 searches, 10 exported clips + 4 timelines,
upscale / RIFE / depth on a 5-s clip); full report in French with every
measurement in `docs/reports/chantier-2-pipeline-apple-silicon.md`.

| M1 Pro 16 GB | Start of round 2 | End of round 2 |
|---|---:|---:|
| **Full pipeline** | 7 min 07 | **4 min 52** |
| — season (indexing + 1045 hover proxies) | 4 min 47 | 2 min 40 |
| — hover proxies alone | 149 s | 18.9 s |
| — 10 exported clips | 11.4 s | 6.2 s |
| — finishing (upscale, RIFE, depth) | 125 s | 122 s |

What changed: hover proxies are cut from a 240p copy that the indexing
decode records with VideoToolbox (no second decode of each episode) and
re-encoded with libx264 superfast — at 240p VideoToolbox is per-frame bound
(7.2 s vs 1.8 s for two passes over 4 min) and the batched pass had
`-realtime 1`, which paces it for live capture; derush exports run 3 at a
time without the 10-s pre-roll decode (frame-identical); SigLIP 2
preprocesses on MPS; depth PNGs are written on 4 threads. Cuts are
identical to the frame on the 3 episodes, embeddings cos ≥ 0.9997, tags
top-5 and motion classes 100 %. Rejected with numbers: GPU letterbox for the
wd-tagger (slower), 32 decoder threads (faster alone, slower in the
pipeline), the temporal depth stabilisation on MPS (23x slower than the CPU).

### Round 3 (tour 3) — media engine, Neural Engine, Core ML

| M1 Pro 16 GB | End of round 2 | **Round 3** |
|---|---:|---:|
| **Full pipeline** | 4 min 52 | **3 min 03** |
| — season (indexing + hover proxies) | 160.2 s | 101.2 s |
| — decode + cuts + motion + streamed tags/embeddings | 138.4 s | 91.8 s |
| — finishing (upscale / RIFE / depth) | 121.6 s (71.0 / 17.6 / 33.0) | 71.4 s (37.3 / 12.9 / 21.3) |

- ffmpeg's VideoToolbox decode is slow *per session* only: 8 sessions in
  parallel decode 4 min of 1080p in 5.6 s with 11 s of CPU (software:
  36 s), HEVC 10-bit 5.7 s vs 9.6 s, bit-identical frames. The indexing pass
  is split into 60-s chunks on 8 sessions and stitched into exactly the
  sequential result (`backend/indexing/chunked.py`).
- The wd-tagger, RIFE and Depth Anything run through Core ML on the GPU in
  worker processes (24.3 vs 17.0 img/s; 91 vs 187 ms per pair, identical;
  123 vs 179 ms per frame, 121 dB); the upscale shares its frames between MPS
  and two Neural Engine processes (71 -> 37 s, 55 dB).
- Hover proxies of an episode are made while the next one indexes; depth
  stabilisation uses an exact sort-based median on 6 threads.
- Cuts identical to the frame on the 3 episodes, embeddings cos >= 0.9997,
  top-5 tags >= 99.8 %, motion classes 100 %, no swap. Details, rejected
  ideas and the RTX 3090 comparison: `docs/reports/chantier-2-pipeline-apple-silicon.md`.

## Functional check on the real episode

- **Cuts**: the detector compared every 4th frame and reported the sampled
  frame, so 310 of the episode's 418 cuts were 1–3 frames late (checked
  against container timestamps) and every exported clip ended with a flash
  of the next shot. Cuts now land on the first frame of the new shot (same
  decisions, all decode paths). Remaining limit: the histogram detector
  misses cuts between shots with similar palettes and some white-flash
  transitions (a few "scenes" hold 2–3 shots; sub-segmentation splits the
  long ones).
- **Search** (thumbnails inspected): "Goku" 12/12 relevant in the top 12,
  "Beerus" 11/12 (one Whis-only shot), "explosion" 12/12, "gros plan visage"
  (French query) 12/12 face/eye close-ups, same for "close-up of a face".
- **Characters**: "Beerus" from 3 example scenes, hand-labelled episode
  (~30–35 of 418 scenes show him). The automatic threshold matched 262
  scenes (precision ~0.11, recall 0.97): within one series every shot looks
  alike to SigLIP. It is now kept above the library's 90th percentile:
  45 matches, precision 0.53–0.64, recall 0.80–0.83. Rejecting Whis shots
  (always next to Beerus) raises the threshold a lot (recall 0.27): lower
  it manually after rejections.
- **Camera motion**: pans, tilts, zooms and shakes checked on frame strips
  are plausible on clean shots; a character crossing a static shot can be
  read as a tilt, and scenes that hide a missed cut get a mixed label.

## Running the benchmark on another machine

```bash
uv sync --extra <cu130|cu126|rocm|mps|cpu> --group dev
uv run python tests/bench_platform.py \
    --clip /path/to/4min-extract.mkv --episode /path/to/full-episode.mkv \
    --data-dir /tmp/amv-bench --out bench.json --md bench.md
```

- Models download on first use (10 MB/s cap); the finishing models must be
  installed from Settings > Models > Finishing models first, or pointed to
  with `AMV_MODELS_DIR`. Missing models or media skip a section with a note.
- `--sections index,search,export,finish,mask` (all by default),
  `--profile accurate|fast|settings`, `--runs N` (indexing repeats).
- `--ref DIR` saves the outputs of the first run (embeddings, tags, cuts,
  finishing frames, mattes) and compares later runs against them (cosine,
  tag agreement, cut agreement, PSNR).
- Each section runs in a child process: a native crash or OOM kill is
  recorded (`crashed`, return code, stderr tail) instead of ending the run.
- Merge repeated runs with `--median-of run1.json run2.json --out med.json`
  and render a before/after table with `--compare before.json after.json`.
- Reported per stage: wall time, frames/s or images/s, peak memory (process
  physical footprint on macOS — it includes Metal buffers that RSS misses —
  plus the Metal/CUDA allocator), system swap-outs, and the backend each
  model really used (ONNX node placement per execution provider, MPS→CPU
  fallback ops).

Method used for the table: machine on AC power with `caffeinate`, no other
job of ours running (macOS background services — Photos analysis, Spotlight
— could not be stopped and were active in both runs), before/after runs
interleaved (before, after, before, after), median of 2 runs.

## NVIDIA — RTX 3090 (24 GB), 6 vCPU VM: full pipeline

`tests/bench_pipeline.py` (3 real episodes indexed as one queue with Search+
and hover proxies, 18 searches, 10 clip exports + 4 timelines, upscale x2 /
RIFE x2 / depth on a 5 s clip), one run per column, warm page cache:

| | before chantier 2 | after | round 3 |
|---|---:|---:|---:|
| full pipeline | 8 min 12 | 5 min 49 (x1.41) | **4 min 32 - 4 min 43** |
| season (indexing + proxies) | 336.6 s | 269.1 s | 193.7 - 203.9 s |
| searches / exports / finishing | 11.8 / 29.6 / 114.0 s | 9.1 / 12.4 / 58.8 s | 7.7 / 12.2 / 59.1 s |
| mean GPU utilisation during the season | 6 % | 19 % | 20-22 % |

Details, per-stage GPU utilisation and quality checks:
[PERFORMANCE-NVIDIA.md](PERFORMANCE-NVIDIA.md) and
[reports/chantier-2-pipeline-nvidia.md](reports/chantier-2-pipeline-nvidia.md).
