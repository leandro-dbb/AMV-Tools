# Changelog

All notable changes to AMV Tools are documented here.

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
