@echo off
setlocal
cd /d "%~dp0"

set "UV_DIR=%~dp0.uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "PATH=%UV_DIR%;%PATH%"

pushd app
where npx >nul 2>nul
if errorlevel 1 (
    echo Node.js / npx not found. Run install.bat first.
    pause
    exit /b 1
)
call npx electron .
popd
