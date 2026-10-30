# Changelog

All notable changes to AMV Tools are documented here.

## 0.5.0-beta.1 — 2026-10-30

First **beta**: installers for Windows, macOS (Apple Silicon) and Linux, still
unsigned. [Release report](docs/reports/release-0.5.0-beta.1.md).

**En bref (pour un monteur)**

- **Envoyer la timeline à son logiciel** : on choisit Premiere Pro, DaVinci
  Resolve, After Effects, Vegas Pro, Final Cut Pro ou « Autre », et l'appli
  écrit le bon format avec les étapes d'import (nouveau : EDL texte pour
  Vegas, script `.jsx` pour After Effects). Imports corrigés (chemins
  Premiere sous Windows, EDL, FCPXML, fréquences d'images mélangées).
- **Synchro musique** (nouvel onglet *Musique*) : on dépose un morceau,
  l'appli trouve tempo, temps, mesures, montées et drops, propose un montage
  calé sur les temps avec les plans indexés (rythme, intensité, personnages,
  verrous, régénérer un passage), puis l'exporte en timeline avec la musique
  et les marqueurs, en marqueurs seuls ou en aperçu MP4.
- **Corrections de bugs** : détection des coupes nettement plus juste ;
  l'indexation d'un MKV HEVC ne bloque plus l'appli ; une indexation
  interrompue reprend où elle s'était arrêtée ; glisser-déposer réparé ;
  export de clip réparé sur les PC Windows sans carte NVIDIA ; noms avec
  accents et japonais conservés ; nombreuses impasses de l'interface
  corrigées (progression lisible, fichiers en échec affichés, menu Finition
  accessible…).
- **Upscale** : AnimeJaNai HD V3 (usage non commercial) remplace l'ancien
  modèle à partir de 8 Go de VRAM et sur Mac : plus fidèle, sans halo sur
  les traits sombres, plus rapide.
- **Profondeur** : Video Depth Anything Large en *Qualité max* sur NVIDIA
  (moins de scintillement, plus de relief).
- **Premier lancement** : le backend proposé suit la carte graphique
  détectée ; les modèles d'analyse se téléchargent avec une progression
  visible.
- Le détourage automatique (Mask Mode) reste désactivé par défaut ;
  l'éditeur de roto n'est pas dans cette version.

**In short (for an editor)**

- **Send the timeline to your editor**: pick Premiere Pro, DaVinci Resolve,
  After Effects, Vegas Pro, Final Cut Pro or Other, and the app writes the
  right format with its import steps (new: Vegas EDL text, After Effects
  `.jsx` script). Import fixes (Premiere paths on Windows, EDL, FCPXML,
  mixed frame rates).
- **Music sync** (new *Music* tab): drop a song, the app finds tempo, beats,
  bars, build-ups and drops, proposes a beat-aligned edit from your indexed
  shots (rhythm, intensity, characters, locks, regenerate a passage), then
  exports it as a timeline with the song and markers, markers only, or an
  MP4 preview.
- **Bug fixes**: much more accurate cut detection; indexing an HEVC MKV no
  longer freezes the app; an interrupted indexing resumes where it stopped;
  drag and drop works again; clip export fixed on Windows PCs without an
  NVIDIA card; accented and Japanese names kept; many UI dead ends fixed
  (readable progress, failed files listed, Finishing menu reachable…).
- **Upscale**: AnimeJaNai HD V3 (non-commercial use) replaces the previous
  model from 8 GB of VRAM and on Mac: more faithful, no halo on dark lines,
  faster.
- **Depth**: Video Depth Anything Large in *Max quality* on NVIDIA (less
  flicker, more relief).
- **First run**: the suggested backend follows the detected GPU; analysis
  models download with visible progress.
- Automatic cutout (Mask Mode) stays off by default; the roto editor is not
  in this release.

### Release review (0.5.0-beta.1)

- macOS: without a certificate the app is now signed **ad hoc**
  (`app/scripts/adhoc-sign.cjs`), so Apple Silicon accepts to run it after
  the quarantine is cleared; the build workflow checks the signature.
- Removed one-off exploration scripts (`app/e2e/explore/`) that depended on
  local media; no personal paths in the reports.

### Bug-fix pass (v06 fix)

([report](docs/reports/v06-fix.md))

- **Cut detection** sees every frame, in context (motion-compensated
  difference on the picture area, local activity, flash / impact-frame
  rejection), one detector shared by the CPU, NVIDIA and Apple paths. On
  frame-accurate annotations of two real extracts (109 cuts): recall
  70 % → 81 % and 38 % → 86 % on the exact frame, precision 72 % → 92 % and
  67 % → 90 %. Apple Silicon's chunked decode is off by default for now.
- **HEVC froze the whole sidecar**: a PyAV log callback (also set by
  torchvision on import) deadlocked the HEVC decoder against the GIL.
  Without hardware decode, indexing now decodes in an ffmpeg subprocess.
  10-bit HEVC in MKV failed the NVIDIA NVDEC pass (wrong bit depth).
- **Interrupted indexing**: the episode shows as *incomplete* (not counted,
  *Resume indexing* button) and resumes where it stopped instead of being
  redone; a tagger error no longer marks an episode indexed without tags.
- **Drag and drop**: the Music drop zone works again (Electron 32+); a file
  dropped outside a zone no longer replaces the app with the file. E2E test.
- Autotune unit test made deterministic (virtual clock).

### Windows "works the first time" + timeline export for every editor (v06 A)

([report](docs/reports/v06-A-windows-timeline.md),
[import guide](docs/import-timeline.md))

- **Windows smoke test on a real VM** (`.github/workflows/windows-smoke.yml`):
  builds the NSIS installer, installs it silently, launches the installed app
  with a user-data path holding spaces and accents, installs the CPU backend,
  indexes, searches, exports a clip and every timeline format, checks the
  degraded paths (offline download message, DirectML/CUDA requested without a
  GPU), quits and checks no process is left. Screenshots and logs as
  artifacts.
- **Fixed on Windows**: clip export failed (500) on every PC without an
  NVIDIA GPU — the bundled ffmpeg lists NVENC/CUDA everywhere; NVENC and CUDA
  decode now need a detected NVIDIA GPU and a failing hardware encode falls
  back to libx264. Sidecar log lines were mojibake (cp1252 pipe). ffmpeg
  error output with Japanese names could not be decoded.
- **First indexing**: SigLIP 2 / wd-tagger are downloaded up front with
  progress in the status bar ("SigLIP 2 base (NaFlex) ↓ 512 / 1500 MB")
  instead of silently inside the first batch (16 min at "0 %" on a slow day).
- **First run**: the setup screen suggests the backend matching the detected
  GPU (CUDA 13, CUDA 12.6 for pre-RTX cards or drivers < 580, DirectML for
  AMD/Intel, CPU) instead of always CUDA 13 (`GET /api/setup/hardware`).
- **Timeline export by editor**: Premiere Pro, DaVinci Resolve, After Effects,
  Vegas Pro, Final Cut Pro or Other, each with its import steps (FR/EN). New
  **Vegas EDL text** (`File → Import → EDL Text File`) and **After Effects
  script** (`.jsx`) exports.
- **Timeline fixes**: FCP 7 `pathurl` spelled like Premiere
  (`file://localhost/C%3a/...`); EDL source/record durations always equal and
  never zero; FROM CLIP NAME / FROM FILE no longer cut at 80 columns; OTIO
  source in points exact for mixed frame rates; FCPXML markers valid against
  Apple's DTD; exports byte-identical on every OS.
- **Validation**: FCPXML against Apple's 1.9 DTD (pinned); every format read
  back by third-party readers and rendered frame by frame with MLT.

### UI journey — the app used end to end like an editor (v06 B)

The built Electron app, real sidecar and real models (CPU) were driven by
Playwright on real anime extracts
([report](docs/reports/v06-B-parcours-ui.md)).

- **E2E suite**: `npm run e2e` (app/e2e/, Playwright `_electron`, own Xvfb,
  throwaway profile, synthetic clips or `AMV_E2E_MEDIA`) covers first launch →
  library + indexing → search → derush basket, clip and timeline exports;
  optional manual CI job `.github/workflows/e2e-linux.yml`.
  `AMV_E2E_SKIP_SYNC=1` runs the first-run install without its `uv sync`.
- **Indexing progress** in the status bar: current stage in plain words (FR/EN),
  file *n/N*, time left, labelled *Stop* button; the sub-segmentation tagger
  pass reports per batch (the bar used to sit at 5 % for 10+ minutes on CPU).
- **Failures are visible**: a file that cannot be indexed (corrupted, still
  copying…) is listed by name with its reason in the Library.
- **Library** follows the running indexing (no more "Ready to index" + active
  start button during a run, list refreshed as episodes finish), queued items
  can be removed, and a first-run note says the analysis models (~2 GB) are
  downloaded when the first indexing starts.
- **Drag and drop of episode folders works again** (Electron 32+ removed
  `File.path`; the preload now exposes `webUtils.getPathForFile`).
- **Hover previews** play from the source file when no proxy exists (the
  default), in Search and Tags; thumbnail fallback when the codec cannot play.
- **Settings > Search** (threshold, max results, sort) is now used by the Search tab.
- **Export folder**: a non-writable folder is refused with a clear message
  instead of failing every later export; settings errors no longer show raw
  `API 400: {json}`.
- First-run screen: no more clipped title at 1280×720, *Fast* profile
  pre-selected and explained for CPU-only installs, right version shown.
- GPU fallback banner: no false "torch is not installed", reasons translated.
- **Scene editor**: the Finishing menu no longer opens under the player (only
  Upscale was reachable), the transport bar stays on screen below ~1000 px.
- **Search**: the threshold slider and sort re-run the search; a hint (and a
  one-click 5 % threshold) when few shots match (French queries find fewer);
  visible wait for the first search (model load); an indexed library no
  longer greeted with "index a folder first"; recent searches updated.
- **Export names keep accented and Japanese letters** ("Clip été 01" used to
  become `Clip_t_01`).
- Derush export summary translated; timeline dialog says *Close* once done;
  the backend-unreachable screen no longer shows a developer command.
- Escape closes the scene editor; a queue left from the previous session is
  explained in the Library at launch.
- Small copy fixes (Derush empty state pointed to Settings instead of Library).

### Music sync (v06 C)

New **Music** tab ([report](docs/reports/v06-C-synchro-musique.md)):

- **Song analysis** (librosa, CPU, cached by content hash): tempo (also
  variable tempo), beats, bars, strong hits, smoothed energy curve, sections
  (intro / verse / build-up / drop / chorus / calm part / outro) and drops.
  On synthetic tracks with exact ground truth: beat F1 0.91-1.0 at 70 ms
  (mean error 5-11 ms), bars 0.90-1.0, 2/2 drops at 6 ms; on two real songs
  annotated by eye 2/4 drops found, no false drop. 1.5-4 s for a 2-4 min
  song on a slow CPU; the analyser warms up when the tab opens.
- **Edit proposal** from the indexed shots: beat-aligned cuts (1-8 beats per
  shot depending on the passage and the chosen rhythm, accelerating through
  build-ups, on strong hits at drops), shots picked by intensity / motion /
  text and character wishes, series filter, no reuse; alternatives, locks,
  regenerate one passage.
- **Exports**: timeline (FCP 7 XML, FCPXML, EDL, OTIO, via the regular
  timeline writers, unchanged) with the song on its own audio track and
  beat / passage markers; markers-only (FCP 7 XML, FCPXML, Resolve marker
  EDL, OTIO, CSV); low-resolution MP4 preview with the music. Real check
  (2 indexed episode extracts + a CC-BY song): cuts 10 ms from the beat on
  average, 21 ms max (frame rounding at 23.976) in the timeline and in the
  rendered MP4.
- **Hidden cuts**: the preview detects shots that contain a cut of the
  episode (missed by the indexer's shot detection), flags them and splits
  them in later proposals (35 -> 7 off-beat cuts on the real check).
- New dependency: `librosa` (ISC).

### Models — non-commercial candidates re-evaluated (chantier 4b)

AMV Tools is non-commercial, so models rejected in chantier 4 only for a
non-commercial license were measured with the same protocol
([report](docs/reports/chantier-4b-modeles-non-commerciaux.md)).

- **Upscale: AnimeJaNai HD V3 UltraCompact** (the-database, CC-BY-NC-SA-4.0)
  replaces 2xHFA2kAVCCompact in every NVIDIA and Mac tier from 8 GB and in
  *Max quality*, and animevideov3 in *Fast* (1.6-2x its throughput): LPIPS
  0.030 vs 0.044 against the real 1080p (PSNR 42.7 vs
  42.5 dB, dark shots 48.7 vs 45.3 dB), no colour halo on dark lines, half
  the convolutions. Downloaded from the official archive, only the 2.5 MB
  checkpoint is kept.
- **Depth: Video Depth Anything Large** (CC-BY-NC-4.0) at 364 px in the
  NVIDIA *Max quality* tier: flicker 0.21 % vs 0.24 % (Small 518 px) and
  visibly sharper relief; 9.3 GB of VRAM, does not fit 24 GB at 518 px.
- **Rejected after measurement**: AnimeJaNai V3.1 (SPAN, ONNX only, no gain
  over V3), GIMM-VFI and AMT (no better than RIFE 4.26 on anime action,
  8-22x slower, AMT needs 14-17 GB at 1080p), Video Depth Anything Base
  (checkerboard artefacts on dark shots), PixAI tagger v1.0 (still no
  Beerus/Whis; finds Asta and Cell but ~50x slower than wd-vit, lower tag F1).
- **Licenses**: the download allow-list now accepts CC-BY-NC-4.0 and
  CC-BY-NC-SA-4.0; Settings > Models (tier table and finishing models) shows a
  *non-commercial use* mark next to those models; the README lists them.
- Finishing registry: a model file can be a member of a checksum-verified zip
  archive (only that member is kept on disk).

## 0.4.0-alpha — 2026-10-25

Four tracks land together: a tested and diagnosable app (Reliability), timeline
hand-off to real NLEs (Send to your editor), finer search (Search+) and local AI
finishing passes (Finishing), followed by per-machine performance work, model
tiers and automatic tuning. Libraries are migrated in place to schema v9.

### Reliability — tests, CI, packaging and diagnostics

- **Backend test-suite (`uv run pytest`)**: pytest + isolated SQLite fixtures,
  synthetic ffmpeg clips with known hard cuts, a FastAPI `TestClient`, and
  mocked SigLIP/wd-tagger models. Covers schema creation/migration from legacy
  v3/v4/v7 databases, cut detection, the indexing queue, mocked search, tag
  aggregation, derush keeps/favorites/folders, real CPU libx264 exports
  (duration checked with `ffprobe`), database relink after a library move,
  status, diagnostics and resilient model downloads.
- **Frontend tests (vitest)**: pure helpers are covered, plus a completeness
  test that fails if any i18n key is missing a French or English string (or if
  `{placeholders}` drift between the two languages).
- **CI**: new `.github/workflows/ci.yml` runs backend pytest on a CPU-only
  Ubuntu runner and `npm ci` + `tsc --noEmit` + `vitest run` + `npm run build`
  on every push/PR.
- **Packaging**: `build.yml` now also builds Linux (`AppImage` + `.deb`) and
  attaches every installer to a **draft** GitHub Release on `v*` tags.
  Windows signing (`CSC_LINK`/`CSC_KEY_PASSWORD`) and macOS notarization
  (`APPLE_ID`/`APPLE_APP_SPECIFIC_PASSWORD`/`APPLE_TEAM_ID`) are wired but
  optional: builds stay unsigned when the secrets are absent. See
  `docs/RELEASING.md`.
- **Diagnostics**: new `GET /api/diagnostics` reports app/Python/Torch/FFmpeg
  versions, requested vs. active acceleration, GPU detection or the CPU fallback
  reason, free disk space, per-model presence/size/load errors and recent
  backend errors. Settings → Diagnostic shows all of it with **Copy report**
  (paste-ready Markdown) and **Open logs folder**. Backend logs now rotate in
  `<data dir>/logs/backend.log`.
- **Explicit GPU fallback**: when CUDA/MPS/DirectML/XPU was requested but the
  app runs on CPU, a non-blocking banner explains why instead of silently
  switching.
- **Resilient model downloads**: retries with exponential backoff, size and
  SHA-256 verification when the Hub exposes them, resumable Hugging Face
  downloads (hf_transfer disabled to keep the 10 MB/s cap), and an actionable
  error with a **Retry download** button in Settings → Diagnostic.
- **Contributor docs**: bug-report/feature-request issue forms and
  `CONTRIBUTING.md`.
- Fixed: CPU-only decode fallback crashed with PyAV ≥ 14 (`av.AVError` was
  renamed `av.FFmpegError`).
- `uv run pytest -m "not slow"` runs the fast suite; `slow` tests run the real
  finishing models on CPU and need their weights.
- The sidecar startup hooks (logging, event loop, Search+ background
  analysis) run from a FastAPI `lifespan` instead of the deprecated
  `on_event("startup")`; startup behaviour is unchanged.
- The fast suite (`uv run pytest -m "not slow"`) runs without any warning
  and no longer hides `DeprecationWarning`s; the finishing ffmpeg
  reader/writer now close their pipes (fd leak per pass in long queues).

### Send to your editor — timeline export

- **New timeline exporters**: FCP 7 XML (xmeml v5), FCPXML 1.9, CMX3600 EDL
  and OpenTimelineIO. Cuts point at the original files with exact frame
  in/out — no transcode.
- Frame-accurate NTSC handling (23.976/29.97/59.94), drop-frame and non-drop
  timecode, mixed source frame rates, VFR approximation with a warning, and
  Unicode/Windows/macOS/Linux path URLs.
- Options: sequence fps/resolution (first clip or explicit), back-to-back or
  N-frame gaps, handles, markers with scene names/tags, source audio toggle.
- “Collect media” copies the originals into `<timeline>_media/` and points
  the timeline at the copies for hand-off to another editor: relative paths
  (`timeline_media/…`) in FCPXML, OTIO and the EDL `FROM FILE` comments;
  absolute `file://` URLs of the copies in FCP 7 XML, whose `<pathurl>`
  importers cannot resolve when relative (see README). Golden files cover
  the collected mode; relative paths of Windows sources no longer depend on
  the host OS and fall back to an absolute URL across drives instead of
  failing.
- New `POST /api/timeline/export`, `POST /api/timeline/preview` and
  `GET /api/timeline/formats` endpoints; “Export timeline” buttons in the
  Derush library, Tags selections and Search results with an FR/EN dialog.
- README “Send To Your Editor” with step-by-step import instructions for
  Premiere, After Effects, Resolve, Final Cut and Vegas.

### Search+ — finer search

- **Camera motion**: every scene is classified as static, pan left/right,
  tilt up/down, zoom in/out or shake, with a magnitude. Lucas-Kanade corner
  tracking + RANSAC similarity fit on the 160×90 frames the cut detector
  already decodes (new optional `frame_sink` hook) — CPU only, no extra
  decode while indexing.
- **Intensity, palette, brightness**: 0–1 intensity score (motion, rapid
  sub-cuts, luminance variation, flashes, short shots), 3-colour palette with
  dominant/accent colour buckets, mean brightness.
- **Background analysis** of libraries indexed before 0.4: starts shortly
  after launch (opt-out: `Settings > Search > Automatically analyse existing
  scenes`, i.e. `searchplus.auto_analyze`) or from
  `Filters > Analyse existing scenes`, yields to indexing, progress in
  the status bar. Stale metrics (merged scenes) are recomputed.
- **Similar shots**: `GET /api/search/similar/{scene_id}` (scene embedding
  neighbours, exclude same episode / same series only, filters), reachable
  from Search cards, Tags rows, Derush (player + library) and the mini editor.
- **Characters** tab: create a character for a series from example scenes,
  dropped images or crops; SigLIP 2 prototype + kNN scoring; validate/reject
  feedback (positive/negative examples) re-scores; auto threshold; saved in
  the library, usable as a search filter and as a query tag.
- **Combined filters** on text/image search and filter-only browse:
  characters (any/all), motion types, intensity min/max, duration min/max,
  dominant colour, series/import group, favorites. Collapsible panel in
  Search; result cards show motion icon, intensity bar, palette and
  character chips.
- **Schema v9** (`_apply_v9`, idempotent, additive): `scene_metrics`,
  `characters`, `character_examples`, `character_matches` + indexes on the
  filter columns and scene duration. v8 libraries open and migrate in place.
- **Search index no longer rescans the library on every query**: the
  in-memory index is reused until something changes it (indexing — per
  finished episode and at the end of the run —, scene merge, database
  merge/cleanup, database switch/import). On the 50k-scene benchmark the
  per-search freshness check drops from ~72 ms to an in-memory lookup;
  plain search 251 → 155 ms, similar shots 113 → 33 ms.

### Finishing — upscale, slow-mo / 60 fps, depth, cutout and Mask Mode reliability

- **Finishing menu** in the mini editor (trimmed clip) and the selects library
  (batch): Upscale, Slow-mo / 60 fps, Depth map, Cutout. Each pass is a
  background job in a FIFO queue (one worker, cooperative cancellation) with
  progress/ETA over `/ws/progress`, a floating queue panel (cancel, reveal
  file), a 5-frame speed estimate and a CPU-only warning. Output goes to
  `<export folder>/finish/` with the export codec; frames are streamed through
  ffmpeg pipes and the source audio window is muxed back.
- **Upscale**: Real-ESRGAN `realesr-animevideov3` (default) and
  `RealESRGAN_x4plus_anime_6B`, x2/x4, tiled inference (auto tile size from
  the GPU) on CUDA/MPS/CPU, audio kept. Archs re-implemented (no `basicsr`).
- **Interpolation**: RIFE 4.25 in native PyTorch (no ONNX: no custom ops, the
  official weights run on CUDA/MPS/CPU directly). x2/x4/x8 slow motion (audio
  muted or time-stretched) or 48/50/60 fps (audio kept); "don't interpolate
  across cuts" duplicates frames at detected hard cuts. Streams the clip pair
  by pair (no whole-clip RAM buffer).
- **Depth**: Depth Anything V2 **Small** (the only Apache-2.0 variant), white =
  near, clip-wide normalisation, per-frame scale/shift alignment and optional
  temporal smoothing; 16-bit PNG sequence, ProRes 422 HQ (10-bit) or H.264,
  frame-synchronous with the clip, for parallax/3D in After Effects.
- **Model management**: Settings > Models > Finishing models — download on
  demand (throttled to 10 MB/s, sha256-verified, pinned revisions), delete,
  size and license shown. Nothing is downloaded at startup; missing weights
  return a clear 409, never a sidecar crash. New `/api/finish/*` endpoints.
- **Mask Mode — measured, then fixed.** A benchmark with exact ground truth
  (`python -m backend.finish.bench_mask`: an anime character composited over
  painted backgrounds — walk, motion-blurred action, low contrast, hard cut —
  16 frames 640x360, ISNet anime model, CPU) showed the model output was good
  but the 0.3 post-processing degraded it:
  - the bidirectional EMA ghosted moving edges (edge error x2);
  - the export-time "BG-aware" alpha re-solve ate cel interiors next to ink
    lines (IoU -2 to -4 points; -6 to -8 combined with the EMA).

  Now: flow-aligned, photometrically validated temporal median instead of the
  EMA; the export edge cleanup only fixes edge *colour* (model alpha kept).

  | clip | IoU ↑ 0.3 → 0.4 | edge error ↓ | flicker (dtSSD) ↓ | composite error ↓ |
  |---|---|---|---|---|
  | walk | 0.922 → **0.984** | 12.07 → **5.99** | 8.91 → **3.24** | 5.73 → **2.62** |
  | low contrast | 0.906 → **0.983** | 14.08 → **6.70** | 9.25 → **3.40** | 6.61 → **2.89** |
  | hard cut | 0.909 → **0.981** | 13.59 → **6.39** | 9.50 → **3.69** | 6.80 → **2.91** |
  | motion blur | 0.658 → 0.669 | **29.76** → 39.46 | 16.15 → **15.23** | **11.97** → 15.18 |

- **Mask Mode — Manual engine (SAM 2)**: prompts on any number of keyframes,
  propagation forward *and* backward (frames before the reference frame used
  to stay empty), soft alpha (`sigmoid(2·logits)`) instead of a hard
  staircase, brush corrections fed back as mask prompts for re-propagation.
  Same benchmark, SAM 2.1 tiny, 8 frames, one click + box on frame 0:

  | clip | pipeline | IoU ↑ | edge error ↓ | flicker ↓ | composite error ↓ |
  |---|---|---|---|---|---|
  | walk | 0.3 | 0.964 | 8.75 | 6.33 | 3.89 |
  | walk | 0.4, same click | 0.964 | **7.58** | **4.91** | **3.30** |
  | walk | 0.4 + one brush fix, re-propagated | **0.973** | **6.26** | **4.42** | **2.86** |
  | hard cut | 0.3 | 0.964 | 8.64 | 6.46 | 4.26 |
  | hard cut | 0.4, same click | 0.965 | **7.55** | **4.96** | **3.65** |
  | hard cut | 0.4 + one brush fix, re-propagated | **0.972** | **6.36** | **4.53** | **3.31** |

  (The brush fix is simulated with the ground-truth matte of the worst frame.
  An extra click keyframe alone did not help on these clips.)
- **Mask Mode UI**: click on any frame (keyframes), add/remove brush with
  size, undo per-frame correction, Re-propagate, checkerboard / solid-colour
  cutout preview (exactly what the export writes).
- Mask Mode **stays disabled by default**: clean shots are now good, but
  motion-blurred action is still poor (IoU 0.67), the default Auto checkpoint
  (BiRefNet general, 444 MB) could not be benchmarked here, the numbers come
  from synthetic clips of a single character, and SAM 2 runs at ~20-40 s/frame
  on CPU. See README > Rotoscope Notes.

### NVIDIA performance — measured on an RTX 3090

Real episodes, `tests/bench_cuda.py`, median of two runs; full tables, method
and rejected ideas in `docs/PERFORMANCE-NVIDIA.md`. Changes that only make
sense on NVIDIA are gated on `backend == "cuda"` (or detected ffmpeg
capabilities); Apple Silicon, CPU and ROCm keep their code paths.

Full pipeline (chantier 2, `tests/bench_pipeline.py`: 3 real episodes
indexed as one queue with Search+ and hover proxies, 18 searches, 10 clip
exports + 4 timelines, upscale/RIFE/depth on a 5 s clip): **5 min 49** on
the RTX 3090 (8 min 12 at the start of the chantier, x1.41). Report:
`docs/reports/chantier-2-pipeline-nvidia.md`.

- **Hover proxies without a second decode**: the indexing NVDEC pass also
  encodes a 240p copy of each video (scale_cuda -> NVENC, source frame rate
  and timestamps) and the proxy pass re-cuts it (h264_cuvid -> NVENC) instead
  of decoding the 1080p episode again: 115.6 s -> 28.2 s for the season.
- **Search+ motion on the GPU**: Lucas-Kanade + RANSAC camera-motion
  estimates run as batched PyTorch (256 sample pairs per block) instead of
  OpenCV in two CPU worker processes; same motion class on 95 % of the
  scenes (`AMV_MOTION_CPU=1` restores the CPU path).
- **Finishing**: RGB -> I420 conversion of upscale / RIFE output frames on
  the GPU (the writer's OpenCV conversion took 181 ms per 4K frame and paced
  the job); depth runs 8 frames per fp16 batch and writes its 16-bit PNGs
  from a small thread pool. 114.0 s -> 58.8 s for upscale x2 + RIFE x2 + depth.
- **Exports**: three NVENC clip exports at a time (29.6 s -> 12.4 s for 10 clips).
- Fixed: **hard cuts 1-3 frames late on NVIDIA** — the single NVDEC pass
  used for indexing never got the "first frame of the shot" refinement of
  the other detectors: 10.3 % of the cuts of the test extract were on the
  shot change, now 87.2 % (same positions as the generic detector).
- Median frames converted NV12 -> RGB on the GPU; the cut detector's
  histogram uses `bincount` (same counts, cuts bit-identical).
- Rejected after measurement: GPU preprocessing for Depth Anything (39.4 dB
  on the final depth, under the 40 dB bar), nvjpeg thumbnails (no faster than
  0.4 ms on the CPU), whole-frame upscale on 24 GB cards (slower than 512 px
  tiles), a 1 MB ffmpeg pipe (no measurable change), reflect-101 borders in
  the GPU tracker (93.7 % same-class scenes instead of 96.2 %).

Round 3 (same benchmark): **4 min 32 - 4 min 43** for the full pipeline (5 min 49 after round 2; 18-26 % faster than the round-2 code in interleaved runs on the same, noisy, shared host). On the
6-vCPU VM the CPU, not the GPU, was the limit (a single pass slows down
x2.6 next to 4 CPU-busy processes, only +10 % next to a GPU load).

- **NVDEC straight into CUDA tensors** (`backend/indexing/nvc_pass.py`,
  optional `PyNvVideoCodec`, MIT): H.264 and HEVC 8/10-bit are decoded into
  device memory and downsized by torch with ffmpeg 5.1 `scale_cuda`'s exact
  arithmetic, so cuts, motion samples and median frames are bit-identical
  to the ffmpeg pass; no per-frame filter graph, download or pipe. HEVC
  episode: 47.9 s / 67.2 CPU-s -> 23.4 s / 26.9 CPU-s. The proxies' 240p copy
  is encoded by NVENC from the GPU frames. Falls back to ffmpeg otherwise.
- Two single passes in flight (the next file the queue needs + the cheapest
  of the next three): NVDEC ~60 % -> ~100 %.
- Fixed: **hover proxies of every scene after a rounded cut were the
  previous scene's** (batched proxy pass since 0.4 round 1, NVENC and
  VideoToolbox): the segment muxer now accepts key frames placed half a
  frame early (`-segment_time_delta`).
- Proxies cut as soon as each video is indexed (low CPU priority), 3 chunks
  for the last one; GPU batches of 64 on 24 GB cards when the batch size is
  left at its default; tagger letterbox, scene palettes and the motion
  tracker's work moved to bigger GPU batches; circular-buffer scanner;
  depth smoothing computes its optical flows on 4 threads.
- Rejected after measurement: decode worker processes, 3 decodes in flight,
  ffmpeg 8.1 (frames not bit-identical to 5.1), PyAV zero-copy NVDEC
  (unsupported), all-intra proxy copies (x3.4 disk), CUDA graphs for the
  tracker (no gain).

### Apple Silicon performance

Measured on a MacBook Pro M1 Pro 16 GB with a real 24-min 1080p episode
(`tests/bench_platform.py`, method and full tables in
`docs/PERFORMANCE.md`). Indexing used to crash on Apple Silicon; an episode
now indexes in 2 min 01 with the default Accurate profile (19 min 25 for the
old code with its crashes worked around) and 43 s with Fast (8 min 52),
without swapping.

- Fixed: **indexing crashed on Apple Silicon** — the wd-tagger's dynamic
  batch made Core ML abort the sidecar (MPSGraph assertion) on the first
  batch. The tagger now runs the same checkpoint on PyTorch MPS (timm,
  fp16, outputs identical to the ONNX reference, 1.5 s load); the Core ML
  path, kept as a fallback, pins the batch so it compiles as one partition.
- Fixed: **corrupted SigLIP 2 embeddings on MPS** — asynchronous
  host-to-device copies raced with freed tensors (30/30 batches corrupted:
  garbage search results or a NaFlex error). Copies are blocking on MPS.
- Fixed: **hard cuts were 1–3 frames late** on every platform (310 of 418
  cuts of the test episode), so exported clips ended with frames of the next
  shot. Cuts now land on the first frame of the new shot; detection
  decisions are unchanged.
- Fixed: a **character's automatic threshold** matched most of an episode
  (262/418 scenes, precision ~0.11 for "Beerus" from 3 scenes): it now stays
  above the library's 90th percentile (45 matches, precision 0.53–0.64,
  recall 0.80–0.83).
- **Decoding**: software decode instead of ffmpeg's VideoToolbox hwaccel,
  which is latency-bound and 2–7x slower on Apple Silicon (VideoToolbox kept
  for HEVC above 1440p); one decode now serves cut detection, motion
  analysis, median frames and sub-segmentation samples, and frames are
  tagged/embedded on the GPU while the CPU keeps decoding.
- **Finishing**: Real-ESRGAN upscale in fp16 on whole frames with the x2
  reduction on the GPU (0.88 → 0.54 s/frame); hover proxies decoded in
  software (0.72 → 0.39 s each).
- **Memory**: the sidecar sets `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25` so the
  MPS allocator recycles its cache before the Mac swaps (SAM 2 Mask Mode:
  13.1 → 7.5 GB, no swap); the tagger and SigLIP stay resident during an
  indexing queue (~1 GB) and are released at its end.
- Settings recommends a batch size for 8 / 16 / 32 GB Macs (it barely
  affects speed there).
- New `tests/bench_platform.py`: the same benchmark on any machine (CUDA,
  ROCm, MPS, CPU) — JSON + Markdown, per-stage timings, peak memory, swap,
  real per-model backend, output quality against a reference run.
- **Full pipeline, round 2** (`tests/bench_pipeline.py`, report in
  `docs/reports/chantier-2-pipeline-apple-silicon.md`): hover proxies no
  longer pace VideoToolbox with `-realtime 1` and are cut from a 240p copy
  recorded by the indexing decode instead of decoding every episode a second
  time and re-encoded with libx264 (season of 3 episodes: 149 → 19 s); full
  pipeline 7 min 07 → 4 min 52 on an M1 Pro; SigLIP 2 preprocesses
  on MPS; derush exports run 3 at a time without the 10-s pre-roll decode
  (same frames, 11.4 → 6.2 s for 10 clips); depth PNG sequences are written
  on 4 threads. Cuts, embeddings, tags and motion classes unchanged on the
  3 episodes.
- **Full pipeline, round 3**: 4 min 52 -> 3 min 03 on an M1 Pro. The
  indexing decode runs as 60-s chunks on 8 parallel VideoToolbox sessions
  (same cuts, medians and motion statistics as the sequential pass, a third
  of the CPU); the wd-tagger, RIFE and Depth Anything run through Core ML on
  the GPU, the x2 upscale shares its frames with two Neural Engine
  processes; hover proxies are made while the next episode indexes. New
  optional dependency of the `mps` extra: coremltools (BSD-3-Clause);
  `AMV_COREML=0`, `AMV_ANE=0`, `AMV_CHUNKED=0` turn each part off.

### Model tiers — the best model for each GPU size

- **Settings > Models > Model tier**: Auto (from the VRAM on NVIDIA, the
  unified memory on a Mac), Fast, 8 / 12 / 16 / 24 GB (Mac: 8 / 16 / 24 /
  32 GB), Max quality, Custom. One table per platform; CPU, DirectML and XPU
  keep the previous defaults. FR/EN.
- Search uses **SigLIP 2 so400m** from 8 GB (Mac: 24 GB) up: +12 % precision@10
  and +10 % nDCG@10 on 25 hand-judged queries over 3 real episodes, French
  queries +21 %, at the same embedding-stage time in the pipeline.
- **Embeddings-only re-index** when the search model changes (cuts, tags and
  keeps untouched, progress, cancel); each library records its embedding
  model and keeps searching with it until re-indexed; vectors of every model
  are kept (switching back is instant).
- Finishing: **2xHFA2kAVCCompact** (CC-BY-4.0, Phhofm) upscaler, much closer
  to the real 1080p than animevideov3 (LPIPS 0.044 vs 0.082) at the same
  speed; **RIFE 4.26** and 4.26 heavy; **Video Depth Anything Small**
  (Apache-2.0, vendored, streamed on the GPU): depth flicker divided by
  4.8 and faster than Depth Anything V2 (26.8 vs 14.8 im/s at 518 px on an
  RTX 3090; 364 px on 8/12 GB cards and on Macs).
- Downloads check the model license against an allow-list first.
- Fixed: wd-swinv2 tagger failed every batch on CUDA; the Core ML tagger
  program was shared between different tagger checkpoints.
- `AMV_VRAM_LIMIT_GB` simulates a smaller GPU; `tests/eval/` holds the
  evaluation harness and the hand judgments. Report:
  `docs/reports/chantier-4-modeles.md`.

### Automatic tuning — the performance settings measured on each machine

- **`backend/autotune/`**: a background micro-bench on first launch and when
  the machine fingerprint changes (GPU/chip, driver, torch, ffmpeg, app
  version), only while nothing is indexing (it stops and retries later if an
  indexing starts). It measures the parallel single-pass decodes on NVIDIA
  (1/2/4), the VideoToolbox sessions of the chunked decode on Apple Silicon
  (2/4/8/12), MPS vs Core ML for the wd-tagger (output checked against MPS),
  and the tagger/SigLIP batch size within the memory tier. Choices are cached
  in `<data dir>/autotune.json`; the former constants (2 decodes, 8 sessions,
  Core ML, the per-VRAM batch) remain the defaults when it has not run or a
  measurement fails, and the existing environment variables keep priority
  (new: `AMV_CHUNK_WORKERS`, `AMV_AUTOTUNE=0`).
- **RTX 3090**: the bench keeps 2 NVDEC decodes (1: 1.63-1.74 s, 2: 1.36-1.51 s,
  4: 1.57-1.66 s per 30 s clip) and batch 64, in 40-45 s.
- **Settings → Diagnostic**: the chosen values, their measures, skipped and
  failed settings, and a *Run the automatic tuning again* button (FR/EN).
  `GET /api/autotune`, `POST /api/autotune/run`.
- `tests/bench_pipeline.py --autotune FILE` seeds a measured `autotune.json`.

### Release fixes — from the pre-release review

- **Timeline export**: FCPXML attribute values escape quotes and apostrophes,
  and FCPXML / FCP 7 drop the control characters XML 1.0 forbids (a clip or
  timeline named `Combat "final"` produced a file no editor would import).
  Markers sit inside their clip in FCPXML and OTIO (they were placed before
  the clip's source in point). OTIO is always written by AMV Tools' own
  writer, which keeps markers and warnings (the `opentimelineio`-library path
  dropped them). *Collect media* never overwrites a different file
  (`name_2.ext`…), copies through a temporary file and removes this run's
  copies if the export fails; the timeline file is written atomically.
  Drop-frame timecode → frames inverse fixed; FCP 7 omits `<labels>` for
  unlabelled clips; a bad frame rate, an unreadable source or a denied write
  answer an explicit 4xx instead of a 500.
- **Sidecar access control**: requests whose `Host` is not `127.0.0.1` /
  `localhost` are refused (DNS rebinding), as are browser `Origin`s other than
  the app's own (including `/ws/progress`). Electron generates a token at each
  launch, hands it to the sidecar through `AMV_TOKEN` and to the renderer
  through the preload; every request carries it (`X-AMV-Token`, or `?token=`
  for images, videos and the websocket). A sidecar started by hand without
  `AMV_TOKEN` keeps working without a token.
- **Installers** run `uv sync --no-dev` (and the app's repair sync `uv run
  --no-dev`): pytest, httpx and opentimelineio are no longer installed on user
  machines.
- **Diagnostic report**: *Copy report* replaces the home folder with `~`, the
  user name with `<user>` and video paths / file names with `<video>.<ext>`
  before the text is pasted into a public issue.
- **Libraries**: switching, removing, cleaning up or merging the primary
  library answers 409 (with the reason shown in the Library tab) while an
  indexing, a re-embedding, a Search+ analysis or another merge runs;
  indexing and re-embedding wait for a merge. A merge source whose path
  contains `#`, `?` or `%` is read correctly and no longer stays locked.
- **Finishing**: depth and cutout jobs are refused with a clear message when
  the clip would need more than half the available RAM (they keep the whole
  clip in memory; the OS used to kill the whole sidecar). The depth
  stabiliser no longer keeps every frame's optical flow. Alpha exports go
  through `<stem>.partial<ext>` and are renamed on success; a failure or a
  cancel (now checked on every frame) leaves no truncated file. The PNG
  sequence `README.txt` credits the depth model actually used. Aborting a
  finishing writer with a full queue no longer leaks its thread. A finishing
  model can't be deleted while it downloads or while jobs are queued (409).
- **Models**: BiRefNet's remote code is pinned to a commit per variant; a
  corrupt Hugging Face download is really fetched again (the retry used to
  relink the same corrupt blob).
- **Search+**: invalid filters answer 422 instead of returning unfiltered
  results; the palettes fall back to the CPU when the GPU batch fails
  (indexing used to drop the video's Search+ metrics); a video re-indexed
  during the background analysis is skipped instead of ending the job.
- **Mask Mode**: brush radius / points and keyframe indexes are bounded.
- **Quitting**: the sidecar no longer hangs while the progress websocket is
  open (it had to be killed by hand); it stops its jobs and child processes
  (ffmpeg, Core ML workers…) in about a second, and Electron kills the whole
  process tree if it is still running after 3 s.
- **macOS**: three parallel clip exports only with VideoToolbox (software
  encoders already fill every core).
- **Schema**: a column migration racing another `init_db` no longer fails;
  opening a library from a newer version logs a warning.
- **CI**: every GitHub Action is pinned to a commit SHA.
- **Docs**: test-machine names, LAN address, NAS paths and raw bench dumps
  removed from `docs/reports`; README finishing table follows the model
  tiers, *Max quality* on Mac keeps wd-vit, RIFE 4.25 lite listed, MatAnyone's
  non-commercial license called out.

## 0.3.0-alpha — 2026-07-18

Full Apple Silicon (M1+) support: every stage that was NVIDIA-only now has a
native macOS hardware path. NVIDIA/CUDA behaviour is byte-for-byte unchanged.

### Added

- **Apple Silicon backend (`mps`)**: new onboarding option "Apple Silicon
  (MPS + Core ML)", recommended by default on macOS. SigLIP 2 + BiRefNet run
  on the Apple GPU via Metal (MPS, float16 — same precision trade-off as
  CUDA), and the unified-memory budget is reported to the batch-size
  recommendation in Settings.
- **SAM 2 roto on macOS**: the `mps` extra installs the PyPI `sam2` package
  on Darwin, enabling the click-to-mask / temporal propagation feature that
  was previously Windows/Linux-only.
- **Core ML execution provider**: wd-tagger and the anime ISNet segmenter now
  run on the Neural Engine/GPU through onnxruntime's CoreML EP (MLProgram
  format, compiled-model cache, graceful CPU fallback + a Settings warning
  when the fallback triggers). This was the single largest indexing
  bottleneck on macOS (pure-CPU tagging).
- **VideoToolbox decode/encode across the whole ffmpeg pipeline**: cut
  detection and median-frame extraction decode on VideoToolbox with the
  `scale_vt` on-device downscale (same architecture as the NVDEC+`scale_cuda`
  path), and hover proxies (including the single-pass batched segment-muxer
  path), scene exports, and the HEVC/10-bit compatibility transcode all
  encode with `h264_videotoolbox`.
- **Hardware H.264 codec family**: the "Match Source — Adaptive High
  Bitrate" export codec now remaps transparently between `h264_nvenc` and
  `h264_videotoolbox` depending on the machine, so a library exported on an
  NVIDIA box re-exports identically on a Mac and vice-versa (libx264 remains
  the universal fallback, same VBR targets).

### Changed

- **macOS build is explicitly arm64** (dmg + zip) with hardened runtime and
  entitlements — signing/notarization-ready once a Developer ID certificate
  is configured. Windows NSIS target is explicitly x64.
- `install.sh` syncs the full `mps` baseline on macOS instead of the CPU-only
  one.
- Model offload and OOM-retry paths now release the MPS allocator cache
  (`torch.mps.empty_cache()`) in addition to CUDA's — important on unified
  memory where the cache pressures the whole system.
- `PYTORCH_ENABLE_MPS_FALLBACK` is set by the backend itself, so manual
  `python backend/server.py` launches behave like Electron launches.
- `uv lock` can now be regenerated on macOS: the dynamic-metadata tensorrt
  sdists (which abort without an NVIDIA driver) are declared statically via
  `[[tool.uv.dependency-metadata]]`.

## 0.2.0-alpha — 2026-07-16

### Added

- **French translation + live language switching**: the entire UI (all tabs,
  settings, onboarding, bootstrap, tutorial, mask mode, mini-editor, derush)
  is now bilingual EN/FR. The language auto-detects from the system locale and
  can be changed instantly from `Settings > Interface > Language` — no restart.
- **macOS packaging + CI**: electron-builder now has a `mac` target (dmg/zip,
  per-platform bundled uv binary) and `.github/workflows/build.yml` builds
  unsigned Windows + macOS artifacts on GitHub-hosted runners (macOS builds
  cannot be produced from Windows). macOS remains untested/experimental.
- **Library tab**: importing/indexing episodes moves out of Settings into a
  new top-level Library tab (next to Search/Tags/Derush) built for first-time
  users: a large drag-and-drop zone with an Add-folder button, an optional
  import group, a single prominent "Start indexing" action (tags-only /
  embeddings-only demoted to advanced links), the episode-folder organiser,
  and databases tucked behind an "advanced" disclosure. The app opens on the
  Library tab when the library is empty, the header database button now leads
  there, and every in-app hint that pointed to Settings → Databases was
  updated. The Databases section is removed from Settings.
- **Relink panel (move library to another PC)**: importing a database whose
  episode files live elsewhere no longer bricks playback/derush. Opening an
  existing database with missing files — or clicking the 🔗 button on any
  database (Settings > Databases) — opens a relink panel that lists missing
  files grouped by their old folder: relink one subfolder at a time (e.g. a
  season), everything at once, or any leftover file individually. Matching is
  by filename; duplicate names are disambiguated by parent folder, true
  ambiguities are reported and left untouched. New endpoint
  `POST /api/databases/relink_folder` (accepts an optional video-id scope).
- **Onboarding questions**: the setup screen now has an English/Français
  toggle (top right) and a "Fast vs Accurate" indexing question — Accurate
  (recommended) enables balanced mode + sub-segmentation, Fast keeps the
  quickest path. Both are changeable later from Settings.
- **VRAM-aware batch size**: `/api/status` now reports the GPU's total VRAM;
  the Batch size setting shows the detected GPU and a recommended value
  (e.g. 12 GB → ≈16) with a one-click Apply button.

### Changed

- Quality-oriented indexing defaults for new installs: mode `balanced` (was
  `fast`) and sub-segmentation ON, so long uncut scenes get split properly.
  Existing installs keep their saved settings.
- The Sub-segmentation toggle now warns when it is ignored because the
  indexing mode is `fast`.

### Fixed

- Startup self-repair: an interrupted first install (network drop, app closed
  mid-download) used to leave a half-built venv that every later launch reused
  as-is (`--no-sync`), crashing forever on `ModuleNotFoundError: uvicorn`. The
  launcher now detects that failure and retries once with a full dependency
  sync, repairing the environment automatically. Sidecar crashes before the
  port opens also surface immediately instead of after a 60s timeout.
- Slow connections no longer abort the first install: uv logs nothing while it
  downloads a single large wheel, and the old 60s silence timeout (plus the
  15 min hard ceiling) killed the bootstrap mid-download on slow links —
  leaving the half-built venv above. While a sync is running the launcher now
  tolerates 10 min of silence and 60 min total; a regular boot from an
  existing venv keeps strict timeouts (2 min / 5 min).

### Derush mode (new)

- New `Derush` tab implementing the AMV-maker dailies workflow on top of the
  indexed scene segmentation:
  - **Player**: plays whole episodes (non-mp4 containers are stream-copied
    into a cached .mp4 on first open — a few seconds, no quality loss; a
    GPU-transcode fallback covers codecs Chromium can't decode), Adobe-style
    **J/K/L** shuttle with 1/2/4/8× speeds in both directions, Space
    play/pause, **M** mute, frame stepping (←/→), scene jumping (↑/↓),
    episode chaining (N/P + auto-advance at the end of an episode).
  - **Keep in one keystroke**: Enter (or the Keep button) toggles the scene
    under the playhead into the derush; the segment strip under the player
    shows every cut and highlights kept ones.
  - **Library**: browse all kept scenes with hover previews, rename them,
    organise them into folders (create/rename/delete), and batch-export
    everything (or one folder) with the regular export settings — files land
    in `exports/derush/<folder>/<custom name>.<ext>` with progress in the UI.
- Schema v5: `derush_folders` + `derush_items` tables (auto-migrated).
- **Merge adjacent scenes**: Ctrl+click segments on the player strip to
  select them, then G (or the Merge button) fuses them into one scene —
  tags, sub-segments and derush keeps are carried over. Useful when the cut
  detector split one shot in two.
- Reworked player shortcuts: **H** keeps the scene (was Enter), **←/→**
  (and ↑/↓) jump between scenes, **,/.** step one frame, and J/L now walk a
  ±1 speed ladder (1×→2×→…→10×, wrapping back to 1×; J below 1× crosses
  into reverse).
- **Import groups**: the index queue takes an optional subfolder name
  ("Pokemon S23"); every episode indexed under it is grouped into a
  collapsible folder in the Derush playlist (schema v6, auto-migrated).
- **Favorites (Ù)**: one level above a keep — Ù toggles the scene into the
  derush flagged for pre-treatment (gold in the strip and library). A plain
  keep upgrades to favorite on Ù; the Library gets a Favorites filter, a
  star toggle per card, and a favorites-only batch export scope.
- **Keep ladder on arrow keys**: ↑ walks the scene up (not kept → kept →
  favorite), ↓ walks it back down (favorite → kept → removed). Scene
  navigation moves to ←/→.
- **Remappable shortcuts**: new Settings > Derush section — click a binding,
  press the new key; conflicts are auto-unbound, and a reset restores the
  defaults. The player and its help bar follow the custom map live.
- **Move episodes between folders**: a per-episode button in the playlist
  reassigns an already-imported episode to any import group (with
  suggestions), or back to ungrouped.
- **Open kept scenes in the editor**: hovering a Library card shows a play
  button that opens the scene in the MiniEditor for playback, trim/cut,
  export or Mask Mode.
- **Multi-folder scenes**: a kept scene can now belong to several Library
  folders at once (e.g. "chill" AND "S23") via a checkbox picker on each
  card; folder filters and per-folder exports see it everywhere it's filed
  (schema v7, legacy single-folder assignments migrated automatically).
- **Episode folders panel**: Settings > Databases gets a visual panel where
  import folders are boxes and episodes are draggable chips — create,
  rename or dissolve folders and drag episodes between them (or back to
  Ungrouped).
- **Premiere-style playhead**: the Derush timeline now has a draggable blue
  playhead — click anywhere to jump there, or hold and drag to scrub
  (playback pauses while scrubbing; seeks are rAF-throttled so fast drags
  stay smooth). Ctrl+click on segments still selects for merge.
- **Timeline zoom (up to ×64)**: Ctrl+wheel zooms around the cursor, plain
  wheel pans when zoomed, +/−/Fit buttons in the transport bar. Makes
  frame-short scenes big enough to Ctrl+click and merge. The view follows
  the playhead during playback and recentres after jumps; dragging the
  playhead against an edge pans the timeline.
- Faster episode chaining: the next episode's remux cache is pre-warmed in
  the background while you watch, so auto-next / N starts instantly.
- The player now distinguishes "source file moved/missing on disk" (with a
  pointer to Settings > Databases relink) from a genuine codec decode
  failure — previously both showed the misleading HEVC message.
- Fixed reverse playback (J below 1×): the shuttle loop could die on a stale
  velocity read, and blind per-frame seeks froze the picture on long-GOP
  h264 — seeks are now chained on the decoder's `seeked` events, anchored to
  the wall clock so the requested speed holds.
- **"Derushed" flag per episode** (schema v8): a check button on each
  playlist row marks an episode as done — green check + dimmed title — and
  each row shows how many scenes you kept in it. Toggleable anytime.

### Export

- New `H.264 NVENC — Match source, high bitrate` codec preset replicating
  Adobe Media Encoder's "Match Source — Adaptive High Bitrate": hardware
  NVENC encode, VBR 1-pass with the target scaled to the source pixel rate
  (15.2 Mbps at 1080p23.976, exactly AME's figure), High profile, full
  Rec.709 VUI signalling, .mp4 container. Falls back to libx264 with the
  same bitrate targets on machines without NVENC.
- Audio re-encode is now AAC 48 kHz stereo with a configurable bitrate
  (192/256/320 kbps, default 320 — AME's default) instead of fixed 192 kbps.

### Roto / Mask Mode overhaul

- Fixed the biggest rendering bug in alpha export: masks were sampled at a
  hard-coded 24 fps while the export decoded the source at its native rate,
  so the matte drifted progressively out of sync on any non-24 fps source
  (23.976/29.97/60 fps, i.e. almost everything). The session now probes the
  real frame rate and the export re-decodes the source through the exact same
  frame sampler, so mask N and frame N always match.
- Replaced the `alphamerge` export path with a unified RGBA-sequence encode
  shared with the decontaminate path.
- Roto frames are now extracted as lossless PNG instead of JPEG (compression
  artefacts degraded matte edges and the colour-based cleanups). SAM 2 still
  receives a JPEG mirror, as its video loader requires it.
- Added guided-filter alpha upsampling at export (`Edge refine`): the low-res
  matte edge is snapped back onto full-resolution line work instead of the
  soft bilinear halo. On by default.
- Added BG-aware edge cleanup at export (on by default): the matte's soft
  band is re-solved against the actual local background via the compositing
  equation (nearest-foreground / nearest-background colour estimates), which
  removes the old-background aura around the character and recovers
  spill-free edge colours. In the synthetic reproduction of the reported
  halo (dark subject on tan floor) it removes 96% of the leaked background
  energy and reduces edge-alpha error vs ground truth from 0.38 to 0.01.
- Added temporal smoothing for the Auto engine (bidirectional EMA with
  hard-cut guard) to remove per-frame silhouette flicker. On by default,
  strength configurable.
- Added an `anime` Auto-model variant: SkyTNT's ISNet anime-seg (ONNX,
  ~170 MB), trained specifically on anime characters — usually better than
  the photo-trained BiRefNet checkpoints on cel-shaded footage.
- Added `opencv-python-headless` as a base dependency: the matte cleanup
  settings (shrink, BG suppression, RGB decontaminate) imported cv2, which
  was not installed — enabling any of them crashed the export.
- Fixed `backend/models/` being excluded from the repository by an unanchored
  `models/` gitignore pattern.

### Earlier

- Added Electron desktop workflow around the FastAPI sidecar.
- Added local scene search, tag browsing, trim preview, and ffmpeg export flows.
- Added experimental roto/alpha export with BiRefNet and SAM 2.
- Added configurable roto resolution and conservative matte cleanup defaults.
- Removed legacy prototype folders and local-only project notes from the public tree.
