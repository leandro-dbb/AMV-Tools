@echo off
setlocal
cd /d "%~dp0"

echo --- AMV Tools bootstrap ---
echo Installing the minimum needed to launch the app.
echo The GPU backend selection happens inside the app on first run.
echo.

set "UV_DIR=%~dp0.uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%UV_DIR%\python"
set "UV_CACHE_DIR=%UV_DIR%\uv_cache"
set "UV_VENV_CLEAR=1"

if not exist "%UV_EXE%" (
    echo Downloading uv into %UV_DIR%...
    if not exist "%UV_DIR%" mkdir "%UV_DIR%"
    powershell -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%UV_DIR%'; $env:UV_UNMANAGED_INSTALL='1'; irm https://astral.sh/uv/install.ps1 | iex"
)

set "PATH=%UV_DIR%;%PATH%"

echo Syncing CPU-only baseline (GPU backend installs in-app)...
"%UV_EXE%" sync --extra cpu
if errorlevel 1 (
    echo.
    echo [!] uv sync failed. Check the message above and re-run install.bat.
    pause
    exit /b 1
)

echo Building the Electron frontend...
pushd app
where npm >nul 2>nul
if errorlevel 1 (
    echo.
    echo [!] Node.js / npm not found in PATH. Install Node 20+ from https://nodejs.org
    popd
    pause
    exit /b 1
)
call npm install --silent
call npm run build
if errorlevel 1 (
    popd
    pause
    exit /b 1
)
popd

set "NAME=AMV Tools"
set "TARGET=amv-tools.bat"
set "ICON=assets\logo\amv-tools-logo.ico"
set "BASE_DIR=%~dp0"
set "TARGET_PATH=%BASE_DIR%%TARGET%"
set "ICON_PATH=%BASE_DIR%%ICON%"
set "SHORTCUT_PATH=%BASE_DIR%%NAME%.lnk"

for /f "delims=" %%i in ('powershell -command "[Environment]::GetFolderPath('Desktop')"') do set "DESKTOP_DIR=%%i"
set "DESKTOP_SHORTCUT_PATH=%DESKTOP_DIR%\%NAME%.lnk"

powershell -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%SHORTCUT_PATH%'); $s.TargetPath='%TARGET_PATH%'; $s.WorkingDirectory='%BASE_DIR%'; $s.IconLocation='%ICON_PATH%'; $s.Save()"
powershell -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%DESKTOP_SHORTCUT_PATH%'); $s.TargetPath='%TARGET_PATH%'; $s.WorkingDirectory='%BASE_DIR%'; $s.IconLocation='%ICON_PATH%'; $s.Save()"

echo.
echo Bootstrap complete. Launch AMV Tools to pick your GPU backend.
exit /b 0
