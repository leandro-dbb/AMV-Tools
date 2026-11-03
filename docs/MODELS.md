# Finishing passes, model tiers and model licenses

[← Back to the README](../README.md)

## Finishing Toolbox — Prepare The Material, Not Just Find It

From the mini editor (on the trimmed clip) or from the selects library (on every
clip shown), the **Finishing** menu runs local AI passes and writes a new file
to `<export folder>/finish/`:

| Pass | Model | What you get |
|---|---|---|
| **Upscale** x2 / x4 | AnimeJaNai HD V3 UltraCompact (per model tier, non-commercial), Real-ESRGAN AnimeVideo v3 (fast, steady) or x4plus anime 6B (sharper) | Upscaled clip with the source audio, using your export codec. Tiled inference keeps VRAM bounded. |
| **Slow-mo / 60 fps** | RIFE 4.26 (or 4.26 heavy, per model tier) | x2/x4/x8 slow motion (audio muted or time-stretched) or 48/50/60 fps at the same duration (audio kept). *Don't interpolate across cuts* duplicates frames at hard cuts instead of morphing two shots. |
| **Depth map** | Video Depth Anything Small (Large in *Max quality* on NVIDIA; Depth Anything V2 Small on 8 GB Macs and the Mac *Fast* tier) | Grayscale depth (white = near), frame-synchronous with the clip, as a 16-bit PNG sequence, ProRes 422 HQ (10-bit) or H.264. Optional temporal smoothing. Made for parallax / 3D layers in After Effects. |
| **Cutout** | Mask Mode's Auto engine | ProRes 4444 / VP9 alpha in one click (requires Mask Mode). |

Every pass is a background job: queue as many as you like, follow progress and
ETA in the floating **Finishing** queue (bottom right), cancel, or reveal the
file. *Estimate time* benchmarks the model on 5 real frames of your clip before
you commit. Without a GPU everything still works on the CPU, just slowly — the
dialog warns you.

Models are **not** bundled: download them on demand from `Settings > Models >
Finishing models` (or from the dialog). Downloads are throttled to 10 MB/s,
checksum-verified, and can be deleted at any time.

| Model | Size | License |
|---|---|---|
| Real-ESRGAN `realesr-animevideov3` | 2.5 MB | BSD-3-Clause |
| Real-ESRGAN `RealESRGAN_x4plus_anime_6B` | 18 MB | BSD-3-Clause |
| `2xHFA2kAVCCompact` by Philip Hofmann ([Phhofm/models](https://github.com/Phhofm/models)) | 4.8 MB | CC-BY-4.0 (attribution: Philip Hofmann) |
| `2x_AnimeJaNai_HD_V3_UltraCompact` by the-database ([mpv-upscale-2x_animejanai](https://github.com/the-database/mpv-upscale-2x_animejanai), official `ModelsOnly` archive, only this file is kept) | 24.7 MB download, 2.5 MB on disk | **CC-BY-NC-SA-4.0 — non-commercial use only** (attribution: the-database) |
| RIFE 4.25 / 4.25 lite / 4.26 / 4.26 heavy (Practical-RIFE, mirrored by vs-rife) | 25 MB each | MIT |
| Depth Anything V2 **Small** | 99 MB | Apache-2.0 (Base/Large are CC-BY-NC and deliberately not offered) |
| Video Depth Anything **Small** (temporally stable, code vendored in `backend/finish/archs/vda`) | 116 MB | Apache-2.0 |
| Video Depth Anything **Large** (same code, *Max quality* tier on NVIDIA) | 1.5 GB | **CC-BY-NC-4.0 — non-commercial use only** |
| Beat This! `small0` (CPJKU; Music tab bar lines — code vendored in `backend/music/beatthis` with rotary-embedding-torch, both MIT) | 8.5 MB | MIT |

**Non-commercial models.** AMV Tools is a free, non-commercial tool, so it
also offers models whose license forbids commercial use (marked *non-commercial
use* in `Settings > Models`). Their weights are never committed or bundled:
they are downloaded on demand from the official source. If you use AMV Tools
or its output commercially, pick a tier or model with a permissive license
(`Custom` + the finishing menu let you choose 2xHFA2kAVCCompact or Video Depth
Anything Small instead).

Which of them a job uses by default follows the **model tier** (below).

RIFE runs as native PyTorch rather than ONNX: the 4.25 network has no custom
ops, loads the official weights directly on CUDA, MPS and CPU, and avoids a
per-resolution ONNX export.

On Apple Silicon the same weights are additionally compiled with Core ML
(`coremltools`, BSD-3-Clause, part of the `mps` extra) the first time a model
meets a new frame size: the upscale shares its frames between the GPU and the
Neural Engine, RIFE, Depth Anything and the wd-tagger run on the GPU through
Core ML (1.4–2x faster than PyTorch MPS). The compiled programs are cached
next to the model files; set `AMV_COREML=0` / `AMV_ANE=0` to stay on MPS.

### Model Tiers — The Best Models For Your GPU

`Settings > Models > Model tier` picks every model (search embeddings,
tagger, upscale, interpolation, depth) for your memory size. **Auto** reads
the VRAM (NVIDIA) or the unified memory (Mac); you can force a tier, *Fast*,
*Max quality* or *Custom* (your own per-model choices). The choices come from
measurements on three real episodes (hand-judged search and tag sets, x2
upscale and frame-interpolation references, depth flicker) on an RTX 3090
and an M1 Pro — see [reports/chantier-4-modeles.md](reports/chantier-4-modeles.md)
and, for the non-commercial models, [reports/chantier-4b-modeles-non-commerciaux.md](reports/chantier-4b-modeles-non-commerciaux.md).

| Tier | Search | Tagger | Upscale | Interpolation | Depth |
|---|---|---|---|---|---|
| Fast | SigLIP 2 base | wd-vit v3 | AnimeJaNai V3 UltraCompact | RIFE 4.26 | Video Depth Anything Small, 364 px (Mac: DA V2 Small, 364 px) |
| NVIDIA 8 / 12 GB | SigLIP 2 so400m | wd-vit v3 | AnimeJaNai V3 UltraCompact | RIFE 4.26 | Video Depth Anything Small, 364 px |
| NVIDIA 16 / 24 GB | SigLIP 2 so400m | wd-vit v3 | AnimeJaNai V3 UltraCompact | RIFE 4.26 | Video Depth Anything Small, 518 px |
| Mac 8 GB | SigLIP 2 base | wd-vit v3 | AnimeJaNai V3 UltraCompact | RIFE 4.26 | DA V2 Small (Core ML) |
| Mac 16 GB | SigLIP 2 base | wd-vit v3 | AnimeJaNai V3 UltraCompact | RIFE 4.26 | Video Depth Anything Small, 364 px |
| Mac 24 / 32 GB+ | SigLIP 2 so400m | wd-vit v3 | AnimeJaNai V3 UltraCompact | RIFE 4.26 | Video Depth Anything Small, 364 px |
| Max quality | SigLIP 2 so400m, 256 patches | wd-eva02-large v3 (Mac: wd-vit v3) | AnimeJaNai V3 UltraCompact | RIFE 4.26 heavy | Video Depth Anything Large, 364 px (Mac: Small, 364 px) |
| CPU / DirectML / XPU | unchanged defaults | | | | |

Changing the search model asks for an **embeddings-only re-index** (cuts,
tags and keeps are kept, progress shown, cancellable); until it is done the
library keeps searching with the model it was indexed with, and every
library remembers its own model. Models download on demand (10 MB/s,
checksum-verified) and only after their license has been checked against an
allow-list (Apache-2.0, MIT, BSD, CC-BY-4.0, CC0, and the non-commercial
CC-BY-NC-4.0 / CC-BY-NC-SA-4.0, flagged in Settings).

| Model | Size | License |
|---|---|---|
| SigLIP 2 base / so400m NaFlex (google) | 1.5 / 4.5 GB | Apache-2.0 |
| wd-vit / wd-eva02-large tagger v3 (SmilingWolf) | 0.4 / 1.3 GB | Apache-2.0 |
| Mask Mode detector: BiRefNet general (ZhengPeng7) | 445 MB | MIT |
| Mask Mode tracker: SAM 2.1 tiny / base+ / large (Meta) | 156 / 324 / 898 MB | Apache-2.0 |
| Mask Mode character pick: SAM 3 (Meta, gated, your own Hugging Face token) | 3.4 GB | SAM License (use and redistribution allowed, military/ITAR uses forbidden) |
| Mask Mode alternatives: ISNet anime-seg (SkyTNT) / ToonOut (joelseytre) | 176 / 885 MB | Apache-2.0 / MIT (dataset CC-BY-4.0) |
| Cut detection network (optional, `Settings > Models`): TransNetV2 (Souček & Lokoč; PyTorch weights Sn4kehead/TransNetV2) | 30 MB | MIT architecture, Apache-2.0 weights |

**Cut detection.** Every frame goes through the hand-made detector; once
TransNetV2 is downloaded (`Settings > Models`, 30 MB) it is combined with
it: on NVIDIA in half precision on its own low-priority CUDA stream (CUDA
graph, frames made on the GPU), on Apple Silicon through Core ML on the
Neural Engine (MPS while Core ML gets ready), on the CPU otherwise. It runs
by itself only where the automatic tuning measured that it costs at most
10 % of the decoding pass; elsewhere (typically CPU-only machines) it is
`Settings > Indexing > Precise cuts (slower)`, off by default, with the
measured extra time per episode. Same cut decisions on every engine,
measured at the exact frame on hand-annotated extracts: see
[reports/coupes.md](reports/coupes.md).

SAM 3 is the one model outside the allow-list: it is not downloaded by AMV
Tools' license-checked downloader but fetched by Hugging Face with *your*
token once you have accepted Meta's SAM License on the model page (the
license text is linked in `Settings > Models`; it allows use, including
commercial, forbids military / nuclear / ITAR uses and reverse engineering).

Mask Mode per tier: BiRefNet general + SAM 2.1 tiny (Fast, NVIDIA 8 GB, Mac
8/16 GB), + base+ (NVIDIA 12/16 GB, Mac 24/32 GB), + large (NVIDIA 24 GB, Max);
SAM 3 character pick on NVIDIA 8 GB and up (not on Fast, not on Mac: not
measured there); ISNet anime without tracking on CPU. The Mac choices are not
measured yet.
