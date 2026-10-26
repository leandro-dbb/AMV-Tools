# End-to-end smoke test of the INSTALLED AMV Tools app on a real Windows VM.
#
# Called by .github/workflows/windows-smoke.yml after the NSIS installer ran
# silently. Everything happens in ONE PowerShell process on purpose: the
# GitHub runner kills the processes a step started when that step ends, so
# the app must be launched, driven and quit within the same step.
#
# What it does, in order (any failure throws and fails the job):
#   1. launch the installed exe with a user-data folder whose path contains
#      spaces and accents (like C:\Users\Jérôme Dupont\AppData\Roaming\…),
#      a fixed port/token (test-only env hooks, see app/electron/main.ts)
#   2. wait for /health (first launch = uv provisions Python + base deps)
#   3. probe GPU detection before setup (no GPU here: must recommend CPU)
#   4. install the CPU backend through the same API the onboarding screen uses
#   5. reload the window and screenshot the main UI
#   6. generate two short test clips with the ffmpeg the app ships
#      (imageio-ffmpeg), with spaces / accents / apostrophes in their names
#   7. queue + index them (CPU: real model download + tagging + embeddings)
#   8. text search, clip export, every timeline format into a folder whose
#      name holds accents and CJK characters, basic checks per format
#   9. offline model download → clear "check your internet" message;
#      DirectML request without torch-directml → clean CPU fallback
#  10. quit through the normal path and check no process is left behind
param(
  [Parameter(Mandatory = $true)] [string] $Exe,
  [Parameter(Mandatory = $true)] [string] $ArtDir
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$port = 8765
$token = 'smoke-' + [guid]::NewGuid().ToString('N')
$base = "http://127.0.0.1:$port"
$headers = @{ 'X-AMV-Token' = $token }

New-Item -ItemType Directory -Force $ArtDir | Out-Null
# Spaces + accents in the user-data path: venv, uv-managed Python, SQLite DB,
# logs and model cache all live under it.
$userData = Join-Path $ArtDir 'Données utilisateur Jérôme'
$shots = Join-Path $ArtDir 'screenshots'
$control = Join-Path $ArtDir 'control'
New-Item -ItemType Directory -Force $userData, $shots, $control | Out-Null
$summary = [ordered]@{}

function Step([string] $name) {
  Write-Host ''
  Write-Host "=== $name ===" -ForegroundColor Cyan
}

function Save-Json($obj, [string] $name) {
  $obj | ConvertTo-Json -Depth 10 | Out-File (Join-Path $ArtDir $name) -Encoding utf8
}

function Api([string] $method, [string] $path, $body = $null, [int] $timeout = 120) {
  $params = @{ Method = $method; Uri = "$base$path"; Headers = $headers; TimeoutSec = $timeout }
  if ($null -ne $body) {
    # Send UTF-8 bytes explicitly: accented paths must reach the sidecar intact.
    $json = $body | ConvertTo-Json -Depth 10
    $params.Body = [Text.Encoding]::UTF8.GetBytes($json)
    $params.ContentType = 'application/json; charset=utf-8'
  }
  try {
    return Invoke-RestMethod @params
  } catch {
    $detail = $_.ErrorDetails.Message
    throw "$method $path failed: $($_.Exception.Message) $detail"
  }
}

function Shot([string] $tag) {
  # The app writes the window capture itself (capturePage) when it sees the
  # control file; we also grab the whole desktop as a fallback.
  New-Item -ItemType File -Force (Join-Path $control "shot-$tag") | Out-Null
  try {
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
    $bmp.Save((Join-Path $shots "desktop-$tag.png"), [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
  } catch {
    Write-Warning "desktop screenshot $tag failed: $($_.Exception.Message)"
  }
}

function Dump-Logs {
  foreach ($f in 'electron.out.log', 'electron.err.log') {
    $p = Join-Path $ArtDir $f
    if (Test-Path $p) { Write-Host "--- tail $f"; Get-Content $p -Tail 60 }
  }
  $bl = Join-Path $userData 'logs\backend.log'
  if (Test-Path $bl) { Write-Host '--- tail backend.log'; Get-Content $bl -Tail 80 }
}

$appProc = $null
try {
  # ── 1. launch ──────────────────────────────────────────────────────────────
  Step 'Launch the installed app'
  $sysFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
  $summary.system_ffmpeg = if ($sysFfmpeg) { $sysFfmpeg.Source } else { $null }
  # The app must not depend on a system ffmpeg: hide any from its PATH.
  $cleanPath = ($env:PATH -split ';' | Where-Object {
      $_ -and -not (Test-Path (Join-Path $_ 'ffmpeg.exe')) }) -join ';'
  $env:PATH_BEFORE_SMOKE = $env:PATH
  $env:PATH = $cleanPath
  $env:AMV_TOKEN = $token
  $env:AMV_SIDECAR_PORT = "$port"
  $env:AMV_SMOKE_CONTROL_DIR = $control
  $env:AMV_SMOKE_SCREENSHOT_DIR = $shots
  $env:ELECTRON_ENABLE_LOGGING = '1'
  $appProc = Start-Process -FilePath $Exe -ArgumentList "--user-data-dir=`"$userData`"" -PassThru `
    -RedirectStandardOutput (Join-Path $ArtDir 'electron.out.log') `
    -RedirectStandardError (Join-Path $ArtDir 'electron.err.log')
  $env:PATH = $env:PATH_BEFORE_SMOKE
  foreach ($v in 'AMV_TOKEN', 'AMV_SIDECAR_PORT', 'AMV_SMOKE_CONTROL_DIR', 'AMV_SMOKE_SCREENSHOT_DIR') {
    Remove-Item "Env:$v"
  }
  Write-Host "App PID $($appProc.Id)"

  # ── 2. health ──────────────────────────────────────────────────────────────
  Step 'Wait for /health (first launch provisions Python)'
  $t0 = Get-Date
  $healthy = $false
  while (((Get-Date) - $t0).TotalMinutes -lt 15) {
    if ($appProc.HasExited) { throw "The app exited early with code $($appProc.ExitCode)" }
    try {
      $h = Invoke-RestMethod -Uri "$base/health" -Headers $headers -TimeoutSec 5
      if ($h.ok) { $healthy = $true; break }
    } catch { Start-Sleep -Seconds 3 }
  }
  if (-not $healthy) { throw 'Sidecar /health never answered within 15 min' }
  $summary.first_boot_s = [int]((Get-Date) - $t0).TotalSeconds
  Write-Host "Sidecar healthy after $($summary.first_boot_s) s"

  # ── 3. GPU probe before setup ──────────────────────────────────────────────
  Step 'GPU detection before setup'
  $hw = Api GET '/api/setup/hardware'
  Save-Json $hw 'hardware.json'
  Write-Host "hardware: $($hw | ConvertTo-Json -Compress -Depth 5)"
  if ($hw.recommended -ne 'cpu') { throw "No GPU on this runner, but setup recommends '$($hw.recommended)'" }
  $summary.recommended_backend = $hw.recommended
  # The onboarding screen runs the same probe: give it time to preselect.
  Start-Sleep -Seconds 4
  Shot 'onboarding'

  # ── 4. CPU backend ─────────────────────────────────────────────────────────
  Step 'Install the CPU backend (same call as the onboarding screen)'
  $r = Api POST '/api/setup/install' @{ backend = 'cpu' } 3600
  $r.log | Out-File (Join-Path $ArtDir 'setup-cpu.log') -Encoding utf8
  if (-not $r.ok) { throw "CPU backend install failed: $($r.hint)`n$($r.log)" }
  $status = Api GET '/api/status'
  Save-Json $status 'status.json'
  Write-Host "status: $($status | ConvertTo-Json -Compress)"
  if ($status.device -ne 'cpu') { throw "Expected device cpu, got '$($status.device)'" }

  # ── 5. main UI ─────────────────────────────────────────────────────────────
  Step 'Reload the window and capture the main UI'
  New-Item -ItemType File -Force (Join-Path $control 'reload') | Out-Null
  Start-Sleep -Seconds 8
  Shot 'main'

  # ── 6. test clips ──────────────────────────────────────────────────────────
  Step 'Generate test clips with the bundled ffmpeg'
  $venvPy = Join-Path $userData '.venv\Scripts\python.exe'
  if (-not (Test-Path $venvPy)) { throw "Sidecar venv not found at $venvPy" }
  $ffmpeg = (& $venvPy -X utf8 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())').Trim()
  if (-not (Test-Path $ffmpeg)) { throw "bundled ffmpeg not found ($ffmpeg)" }
  $summary.bundled_ffmpeg = $ffmpeg
  $media = Join-Path $ArtDir "médias d'essai"
  New-Item -ItemType Directory -Force $media | Out-Null
  $clipA = Join-Path $media 'DBS Beerus 01 — extrait é.mp4'
  $clipB = Join-Path $media "Black Clover S02E04 l'épisode.mkv"
  # A: 23.976 fps with hard cuts between colour cards (scene detection must
  # find several scenes) + audio. B: 25 fps, no audio.
  $cards = @('testsrc2', 'smptebars', 'rgbtestsrc', 'testsrc')
  $inputs = @(); $filters = @()
  for ($i = 0; $i -lt $cards.Count; $i++) {
    $inputs += @('-f', 'lavfi', '-t', '6', '-i', "$($cards[$i])=size=1280x720:rate=24000/1001")
    $filters += "[$($i):v]"
  }
  $concat = ($filters -join '') + "concat=n=$($cards.Count):v=1:a=0[v]"
  & $ffmpeg -y -hide_banner -loglevel error @inputs -f lavfi -t 24 -i 'sine=frequency=440:sample_rate=48000' `
    -filter_complex $concat -map '[v]' -map "$($cards.Count):a" -c:v libx264 -preset ultrafast -pix_fmt yuv420p `
    -c:a aac -shortest $clipA
  if ($LASTEXITCODE -ne 0) { throw 'ffmpeg failed for clip A' }
  & $ffmpeg -y -hide_banner -loglevel error -f lavfi -t 6 -i 'mandelbrot=size=640x360:rate=25' `
    -f lavfi -t 6 -i 'testsrc2=size=640x360:rate=25' -filter_complex '[0:v][1:v]concat=n=2:v=1:a=0[v]' `
    -map '[v]' -c:v libx264 -preset ultrafast -pix_fmt yuv420p $clipB
  if ($LASTEXITCODE -ne 0) { throw 'ffmpeg failed for clip B' }
  Get-ChildItem $media | Format-Table Name, Length

  # ── 7. index ───────────────────────────────────────────────────────────────
  Step 'Queue and index both clips on CPU'
  $q = Api POST '/api/index/queue' @{ paths = @(
      @{ path = $clipA; recursive = $false; group = 'Smoke é' },
      @{ path = $clipB; recursive = $false; group = 'Smoke é' }) }
  if ($q.added -lt 2) { throw "Queue accepted $($q.added) of 2 clips" }
  Api POST '/api/index/start' @{ phases = @('tag', 'embed') } | Out-Null
  $t0 = Get-Date
  $done = $false
  while (((Get-Date) - $t0).TotalMinutes -lt 30) {
    $videos = @((Api GET '/api/videos').videos)
    $line = ($videos | ForEach-Object { "$($_.display_name)=$($_.status)/$($_.scene_count)" }) -join '; '
    Write-Host "[$([int]((Get-Date) - $t0).TotalSeconds) s] $line"
    $failed = @($videos | Where-Object { $_.status -in @('failed', 'error') })
    if ($failed.Count) { throw "Indexing failed: $line" }
    $bl = Join-Path $userData 'logs\backend.log'
    if ((Test-Path $bl) -and (Select-String -LiteralPath $bl -Pattern 'Traceback \(most recent call last\)' -Quiet)) {
      throw "A traceback appeared in backend.log while indexing ($line)"
    }
    $ok = @($videos | Where-Object { $_.scene_count -gt 0 -and $_.status -eq 'completed' })
    if ($videos.Count -ge 2 -and $ok.Count -ge 2) { $done = $true; break }
    Start-Sleep -Seconds 10
  }
  if (-not $done) { throw 'Indexing did not finish within 30 minutes' }
  $summary.index_s = [int]((Get-Date) - $t0).TotalSeconds
  Save-Json $videos 'videos.json'
  Save-Json (Api GET '/api/models/status') 'models-after.json'

  # ── 8. search + exports ────────────────────────────────────────────────────
  Step 'Search'
  $search = Api POST '/api/search' @{ query = 'colorful test pattern'; top_k = 10; threshold = 0.0 } 600
  Save-Json $search 'search.json'
  if ($search.total -lt 1) { throw 'Search returned no results' }
  $summary.search_results = $search.total

  Step 'Clip export'
  $sceneIds = @()
  foreach ($v in $videos) {
    $scenes = @((Api GET "/api/videos/$($v.id)/scenes").scenes)
    $sceneIds += @($scenes | Select-Object -First 3 | ForEach-Object { $_.id })
  }
  if ($sceneIds.Count -lt 3) { throw "Expected at least 3 scenes, got $($sceneIds.Count)" }
  $out = Join-Path $ArtDir 'exports é 日本'
  New-Item -ItemType Directory -Force $out | Out-Null
  $clipOut = Join-Path $out "clip l'été.mp4"
  $clip = Api POST '/api/export' @{ scene_id = $sceneIds[0]; output_path = $clipOut } 600
  if (-not (Test-Path -LiteralPath $clip.output) -or (Get-Item -LiteralPath $clip.output).Length -lt 1000) {
    throw "Clip export failed: $($clip.output)"
  }
  $summary.clip_export = (Get-Item -LiteralPath $clip.output).Length

  Step 'Timeline export, every format'
  $formats = @((Api GET '/api/timeline/formats').formats)
  Save-Json $formats 'formats.json'
  $ids = @($formats | ForEach-Object { $_.id })
  foreach ($need in 'fcp7xml', 'fcpxml', 'edl', 'otio', 'vegas', 'aftereffects') {
    if ($ids -notcontains $need) { throw "Missing timeline format $need (got $($ids -join ', '))" }
  }
  $summary.timeline = [ordered]@{}
  foreach ($f in $formats) {
    $target = Join-Path $out ("timeline $($f.id)$($f.extension)")
    $r = Api POST '/api/timeline/export' @{
      scene_ids = $sceneIds; format = $f.id; output_path = $target
      options = @{ sequence_fps = 'first'; layout = 'gaps'; gap_frames = 2; markers = $true
                   include_audio = $true; title = "Smoke é $($f.id)" } } 300
    if (-not (Test-Path -LiteralPath $target)) { throw "Timeline $($f.id) was not written" }
    $text = [IO.File]::ReadAllText($target, [Text.Encoding]::UTF8)
    switch ($f.id) {
      'fcp7xml' { $x = [xml]$text; if (-not $x.xmeml) { throw 'fcp7xml: no xmeml root' }
                  if ($text -notmatch 'file://localhost/[A-Za-z]%3a/') { throw 'fcp7xml: pathurl is not a Premiere-style Windows file URL' } }
      'fcpxml' { $x = [xml]$text; if (-not $x.fcpxml) { throw 'fcpxml: no fcpxml root' }
                 if ($text -notmatch 'src="file:///[A-Za-z]:/') { throw 'fcpxml: src is not a Windows file URL' } }
      'otio' { $j = $text | ConvertFrom-Json; if ($j.OTIO_SCHEMA -notlike 'Timeline.*') { throw 'otio: bad schema' } }
      'edl' { if ($text -notmatch 'FROM CLIP NAME:') { throw 'edl: no FROM CLIP NAME' } }
      'vegas' { if (-not $text.StartsWith('"ID";"Track";"StartTime"')) { throw 'vegas: not a Vegas EDL text header' }
                if ($text -notmatch ';VIDEO;"[A-Za-z]:\\') { throw 'vegas: no VIDEO event with a Windows path' }
                if ($text.Contains("`r`r")) { throw 'vegas: doubled CR (newline translation bug)' } }
      'aftereffects' { if ($text -notmatch 'app\.project') { throw 'aftereffects: not an AE script' } }
    }
    $summary.timeline[$f.id] = "$($r.clip_count) clips, $((Get-Item -LiteralPath $target).Length) bytes"
    Write-Host "  $($f.id): $($summary.timeline[$f.id])"
  }

  # ── 9. degradation checks inside the app's own venv ────────────────────────
  Step 'Offline download message + DirectML fallback'
  $resources = Join-Path (Split-Path $Exe) 'resources'
  $check = Join-Path $ArtDir 'degradation-check.py'
  @'
import os, sys
from backend.models.device import detect_device
auto = detect_device("auto")
dml = detect_device("dml")
cuda = detect_device("cuda")
print("auto:", auto.backend, auto.name)
print("dml requested:", dml.backend, dml.name)
print("cuda requested:", cuda.backend, cuda.name)
if {auto.backend, dml.backend, cuda.backend} != {"cpu"}:
    sys.exit(2)
from backend.model_download import ModelDownloadError, download_file
try:
    download_file("hf-internal-testing/tiny-random-gpt2", "config.json", max_attempts=2, base_delay=0.1)
except ModelDownloadError as exc:
    print("offline message:", exc)
    if "internet" not in str(exc).lower():
        sys.exit(3)
else:
    print("UNEXPECTED: download worked with an unreachable endpoint")
    sys.exit(4)
print("OK")
'@ | Out-File $check -Encoding utf8
  Push-Location $resources
  $env:HF_ENDPOINT = 'http://127.0.0.1:9'
  $env:HF_HUB_OFFLINE = '0'
  $env:HF_HOME = Join-Path $ArtDir 'hf-offline'
  $env:AMV_DATA_DIR = Join-Path $ArtDir 'offline-data'
  $env:PYTHONPATH = $resources   # the script lives elsewhere: sys.path[0] is not the cwd
  & $venvPy -X utf8 $check 2>&1 | Tee-Object -FilePath (Join-Path $ArtDir 'degradation-check.log')
  $rc = $LASTEXITCODE
  foreach ($v in 'HF_ENDPOINT', 'HF_HUB_OFFLINE', 'HF_HOME', 'AMV_DATA_DIR', 'PYTHONPATH') { Remove-Item "Env:$v" }
  Pop-Location
  if ($rc -ne 0) { throw "Degradation check failed (exit $rc)" }

  Shot 'end'
  Start-Sleep -Seconds 3

  # ── 10. quit ───────────────────────────────────────────────────────────────
  Step 'Quit through the normal path'
  New-Item -ItemType File -Force (Join-Path $control 'quit') | Out-Null
  $t0 = Get-Date
  while (-not $appProc.HasExited -and ((Get-Date) - $t0).TotalSeconds -lt 60) { Start-Sleep -Seconds 1 }
  if (-not $appProc.HasExited) { throw 'The app did not quit within 60 s of the quit request' }
  $summary.quit_s = [int]((Get-Date) - $t0).TotalSeconds
  Start-Sleep -Seconds 3
  $installDir = Split-Path $Exe
  $left = @(Get-CimInstance Win32_Process | Where-Object {
      ($_.ExecutablePath -and ($_.ExecutablePath.StartsWith($installDir) -or $_.ExecutablePath.StartsWith($userData))) -or
      ($_.CommandLine -and $_.CommandLine.Contains($userData)) })
  if ($left.Count) {
    $left | Select-Object ProcessId, Name, CommandLine | Format-List | Out-String | Write-Host
    throw "$($left.Count) process(es) left running after quit"
  }
  Write-Host 'No process left behind.'

  # The sidecar's own log must be clean.
  $bl = Join-Path $userData 'logs\backend.log'
  if (Test-Path $bl) {
    # The log is UTF-8: '—' must not have become mojibake on the way.
    if (Select-String -LiteralPath $bl -Pattern ([char]0xFFFD) -SimpleMatch -Quiet) { throw 'backend.log contains U+FFFD (encoding bug)' }
    $bad = @(Select-String -LiteralPath $bl -Pattern 'Traceback \(most recent call last\)|ModuleNotFoundError|UnicodeEncodeError|UnicodeDecodeError|PermissionError')
    if ($bad.Count) { $bad | ForEach-Object { Write-Host $_.Line }; throw "backend.log has $($bad.Count) error line(s)" }
  }
  $summary.result = 'PASS'
} catch {
  $summary.result = 'FAIL'
  $summary.error = "$_"
  Write-Host "FAILED: $_" -ForegroundColor Red
  Dump-Logs
  throw
} finally {
  Save-Json $summary 'summary.json'
  if ($appProc -and -not $appProc.HasExited) {
    Stop-Process -Id $appProc.Id -Force -ErrorAction SilentlyContinue
    & taskkill /F /T /PID $appProc.Id 2>$null | Out-Null
  }
  $bl = Join-Path $userData 'logs\backend.log'
  if (Test-Path $bl) { Copy-Item $bl (Join-Path $ArtDir 'backend.log') -Force }
}
