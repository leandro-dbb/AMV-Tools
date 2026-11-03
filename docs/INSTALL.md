# Install, hardware and troubleshooting

[← Back to the README](../README.md)

## Install On Windows

The recommended path for end users is the Windows installer published on the
project's GitHub Releases page.

1. If you have an NVIDIA GPU, make sure your driver is at least `581.x` for
   CUDA 13. Run `nvidia-smi` in PowerShell to confirm.
2. Download `AMV.Tools.Setup.<version>.exe` from the [releases page](https://github.com/leandro-dbb/AMV-Tools/releases).
3. Run it. The installer is currently unsigned, so Windows SmartScreen may
   warn you. Choose `More info -> Run anyway`.
4. On first launch, pick the backend that matches your hardware, your indexing
   preference (fast/accurate) and your language. The app downloads the chosen
   PyTorch backend on demand: about 4.5-5 GB for NVIDIA CUDA 13 (measured), a
   few hundred MB for CPU — a slow connection is fine, the
   bootstrap waits for it.

> [!IMPORTANT]
> **Smart App Control** (enabled by default on some Windows 11 machines) blocks
> unsigned installers outright, with no "run anyway" option. If nothing happens
> when you launch the installer, check `Windows Security > App & browser
> control > Smart App Control`. Disabling it is permanent; the alternative is
> building from source, which is unaffected.

The installer bundles `uv.exe` and the FastAPI sidecar. Model weights and
PyTorch wheels are downloaded into the user data directory when needed. If a
first install is interrupted (network drop, closed app), the next launch
detects the broken environment and repairs it automatically.

### macOS and Linux

The release also carries a macOS build (Apple Silicon, `.dmg` / `.zip`) and
Linux builds (`.AppImage` / `.deb`, x64):

- **macOS**: since 0.5.0-beta.4 the app is signed with a Developer ID and
  notarized by Apple: drag it into Applications and open it. Older builds
  (0.5.0-beta.1 to beta.3) were signed ad hoc only: on macOS 15+ the first
  double-click shows *“AMV Tools” Not Opened* — click **Done** (the default
  button, *Move to Trash*, deletes the app), then System Settings → Privacy &
  Security → **Open Anyway**; or, before the first launch,
  `xattr -dr com.apple.quarantine "/Applications/AMV Tools.app"`.
- **Linux**: `chmod +x` the AppImage and run it. It needs `libfuse2` (on
  Ubuntu 24.04 `libfuse2t64`) and `fusermount` (package `fuse3`), present on
  most desktops; without the latter it stops at once on *fuse: failed to exec
  fusermount* (`sudo apt install fuse3`). Or install the `.deb`
  (`sudo apt install ./amv-tools_<version>_amd64.deb`), which needs neither.

On Apple Silicon (`--extra mps`) indexing, search, export,
finishing and Mask Mode have been benchmarked and checked end-to-end on an
M1 Pro 16 GB (a 24-min episode indexes in about 2 minutes with the default
Accurate profile, ~43 s with Fast — see [PERFORMANCE.md](PERFORMANCE.md)),
and the packaged app was tested on a MacBook M1 Pro under macOS 26
([reports/test-installeur-mac.md](reports/test-installeur-mac.md)). On Macs
the wd-tagger runs on the GPU through PyTorch and downloads the same
checkpoint as `model.safetensors` (Apache-2.0) instead of `model.onnx`.
The Linux `AppImage` and `.deb` were checked end-to-end on Debian 12 with an
RTX 3090 (install, CUDA 13 engine, indexing, search, exports, music sync,
finishing, cut-out editor, uninstall — see
[reports/test-installeurs-linux-mac.md](reports/test-installeurs-linux-mac.md));
not yet on Ubuntu, Fedora or a Wayland session. You can also
build from source ([DEVELOPMENT.md](DEVELOPMENT.md)) — `.github/workflows/build.yml`
produces macOS builds on a Mac runner and Linux packages on Ubuntu (macOS
builds cannot be produced from Windows). See [RELEASING.md](RELEASING.md) for the
signing/notarization secrets and the draft-release flow.

### Moving A Library To Another PC

Copy your `.db` file, install AMV Tools on the new machine, then
`Library > Databases (advanced) > Open existing`. If episode files are not at
their old location, the relink panel opens automatically: point it at the
folder that now contains your episodes (subfolders are scanned recursively),
relink season by season, or fix leftovers file by file. Scenes, tags,
embeddings and derush keeps all survive — no re-indexing.

### Diagnostics And Logs

If something misbehaves, open `Settings > Diagnostic`. It shows the app, Python,
PyTorch and FFmpeg versions, the requested vs. actually active acceleration
backend, the detected GPU (or the reason the app fell back to CPU), free disk
space, the state of every model (present/absent, size, load error) and recent
backend errors. **Copy report** puts a Markdown summary on the clipboard, ready
to paste into a bug report; **Open logs folder** reveals the rotating backend
log (`backend.log`) in the user data folder.

When a GPU backend was selected but is unavailable, a non-blocking banner at the
top of the app says so instead of silently running on CPU. Model downloads
retry with backoff and verify size/hash when Hugging Face exposes them; a failed
download shows the reason plus a **Retry download** button.

### Clean Reinstall

If the local environment gets stuck, use `Settings > Advanced > Reinstall from
scratch` inside the app, or wipe manually:

```powershell
Get-Process | Where-Object { $_.ProcessName -like "AMV Tools*" } | Stop-Process -Force -ErrorAction SilentlyContinue
$uninstaller = "$env:LOCALAPPDATA\Programs\amv-tools\Uninstall AMV Tools.exe"
if (Test-Path $uninstaller) { Start-Process -FilePath $uninstaller -ArgumentList "/S" -Wait }
Remove-Item "$env:LOCALAPPDATA\Programs\amv-tools" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:APPDATA\amv-tools" -Recurse -Force -ErrorAction SilentlyContinue
```

The Hugging Face model cache under `%USERPROFILE%\.cache\huggingface` is
preserved. Deleting it forces a multi-GB model re-download on next launch.

macOS: quit the app, then delete `AMV Tools.app` (from Applications or
`~/Applications`), `~/Library/Application Support/amv-tools` (Python,
packages, database, finishing models, exports) and
`~/Library/Preferences/com.amvtools.app.plist`. The search, tagging and
masking models (SigLIP 2, wd-tagger, BiRefNet, SAM 2, TransNetV2: about
2.5 GB after a full test) are in the shared Hugging Face cache,
`~/.cache/huggingface/hub/models--*`; delete those folders too to free the
space (other apps may use that cache).

## Hardware Support

AMV Tools is developed and tested primarily on NVIDIA GPUs (RTX-class, CUDA 12
or 13). That is the recommended path for both indexing and the roto/alpha
pipeline. For any other configuration, CPU mode is the safest fallback — it
works everywhere but indexing is much slower.

The batch size setting auto-recommends a value based on your detected VRAM
(e.g. 12 GB on NVIDIA → 32); lower it if you hit out-of-memory errors.

DirectML, ROCm, and XPU backends are wired into the installer, but they have
not been validated end-to-end on real AMD or Intel hardware. If you are on AMD
or Intel, expect CPU mode to be the reliable path. Apple Silicon (MPS) is
measured in [PERFORMANCE.md](PERFORMANCE.md); run
`tests/bench_platform.py` on your machine to compare.

### NVIDIA Driver Notes

The CUDA 13 backend (`cu130`, recommended for RTX 20xx and newer) needs a
recent NVIDIA driver. If the driver is too old, PyTorch can silently fall back
to CPU and indexing becomes much slower.

| Backend | Minimum driver | Recommended |
| --- | --- | --- |
| `cu130` | 581.x | Latest Game Ready / Studio |
| `cu130-trt` | 581.x | Latest Game Ready / Studio |
| `cu126` | 555.x | 560+ |

Check your driver with `nvidia-smi`. If you cannot update it, choose
`NVIDIA CUDA 12.6` during onboarding instead of CUDA 13.

### NVIDIA Performance

On an RTX 3090 a 23-minute 1080p episode indexes in well under a minute:
NVDEC decodes straight into CUDA tensors (optional `PyNvVideoCodec`, MIT,
installed with the CUDA extras; ffmpeg is used without it), torch downsizes
the frames bit-identically to ffmpeg's `scale_cuda`, and one pass feeds cut
detection, Search+ motion analysis, the scene frames and the hover proxies;
two episodes of a queue decode at once. The whole pipeline benchmark
(3 episodes indexed with proxies, searches, 10 exports, upscale/RIFE/depth)
takes 4 min 32 - 4 min 43. Measurements, method and recommended settings:
[PERFORMANCE-NVIDIA.md](PERFORMANCE-NVIDIA.md).

- **Linux**: the bundled ffmpeg has no NVDEC/NVENC. AMV Tools picks a system
  ffmpeg that has them (`sudo apt install ffmpeg` on Debian/Ubuntu); point
  `AMV_FFMPEG` at another build if needed. Diagnostics shows which one is used.
- **Finishing and exports**: choose *H.264 Hardware (NVENC)* in Settings →
  Export. It matches or beats x264 CRF 18 in quality (larger files) and makes
  upscale / slow-motion jobs 2-3× faster, because x264 is the bottleneck there.
- **Batch size**: the Settings recommendation is calibrated per VRAM
  (6 / 8 / 12 / 24 GB → 8 / 16 / 32 / 64); the default models stay near 1 GB
  of VRAM even at 64, so 8 GB cards are safe.

### Automatic Tuning

The decode and model settings above were measured on two machines (RTX 3090,
M1 Pro). On first launch — and again when the GPU or chip, the driver, PyTorch,
FFmpeg or the app version changes — AMV Tools measures them on yours, in the
background while nothing is indexing (~40 s on the RTX 3090, on 10 s of a
library video, or a synthetic clip when the library is empty):

- parallel decodes: NVDEC single passes in flight (1 / 2 / 4) on NVIDIA,
  VideoToolbox sessions (2 / 4 / 8 / 12) on Apple Silicon — the smallest value
  within 5 % of the fastest is kept;
- Apple Silicon: MPS vs Core ML for the tagger (Core ML only if its tags match
  and it is ≥ 5 % faster); NVIDIA: TensorRT is reported but not used (no
  compatible model path);
- the tagger/SigLIP batch size, never above the memory tier's value.

Settings → Diagnostic shows the choices with their measures and a **Run the
automatic tuning again** button. The results live in `autotune.json` in the
data folder; without them the previous constants apply. `AMV_PARALLEL_DECODES`,
`AMV_CHUNK_WORKERS`, `AMV_COREML=0` and a batch size changed in Settings still
win; `AMV_AUTOTUNE=0` turns the background run off.
