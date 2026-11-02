"""Microsoft Visual C++ runtime on Windows (needed by PyTorch).

torch's DLLs (c10.dll…) link against msvcp140.dll / vcruntime140_1.dll.
The Python that uv installs only ships vcruntime140(_1).dll, and a fresh
Windows 11 has no Visual C++ Redistributable: `import torch` then dies with
"WinError 126 … c10.dll or one of its dependencies" (seen on a clean
Windows 11 26H2 + RTX 3090, 0.5.0-beta.2). The first-run setup checks for
the runtime and, when it is missing, installs Microsoft's official
redistributable (Windows asks the user for permission once).
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
RUNTIME_DLLS = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
# vc_redist exit codes that mean "the runtime is there now".
_OK_CODES = {0, 1638, 3010}   # success, a newer version is installed, reboot pending
_CANCELLED = 1223             # the user refused the Windows permission prompt

MISSING_HINT = (
    "The Microsoft Visual C++ Redistributable (x64) is missing on this PC: PyTorch "
    f"cannot load without it. Install it from {VC_REDIST_URL} (accept the Windows "
    "permission prompt), then retry the same backend."
)


def missing_dlls(loader: Callable[[str], object] | None = None) -> list[str]:
    """Runtime DLLs Windows cannot load (empty elsewhere / when all present)."""
    if sys.platform != "win32" and loader is None:
        return []
    load = loader or (lambda name: ctypes.WinDLL(name))  # type: ignore[attr-defined]
    missing = []
    for name in RUNTIME_DLLS:
        try:
            load(name)
        except OSError:
            missing.append(name)
    return missing


def looks_like_missing_runtime(log: str) -> bool:
    low = log.lower()
    return ("visual c++ redistributable is not installed" in low
            or ("winerror 126" in low and ".dll" in low))


def _run_elevated(exe: Path) -> int:
    """Runs the installer through UAC (Start-Process -Verb RunAs) and waits."""
    # The path goes inside the command (single-quoted): with -Command, extra
    # arguments are appended to the script text, they never reach $args.
    quoted = "'" + str(exe).replace("'", "''") + "'"
    ps = (f"$p = Start-Process -FilePath {quoted} -ArgumentList '/install','/passive','/norestart' "
          "-Verb RunAs -Wait -PassThru; exit $p.ExitCode")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined]
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=flags,
    )
    err = (proc.stderr or "").lower()
    # A refused prompt makes Start-Process throw ("…annulée par l'utilisateur").
    if proc.returncode != 0 and ("cancel" in err or "annul" in err):
        return _CANCELLED
    return proc.returncode


def _download(url: str, dest: Path, log: Callable[[str], None]) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done, next_pct = 0, 25
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total and done * 100 // total >= next_pct:
                log(f"  {done * 100 // total} % of {total // 2**20} MB\n")
                next_pct += 25
    tmp.replace(dest)


def ensure_msvc_runtime(
    cache_dir: Path,
    log: Callable[[str], None],
    *,
    loader: Callable[[str], object] | None = None,
    download: Optional[Callable[[str, Path, Callable[[str], None]], None]] = None,
    run_elevated: Optional[Callable[[Path], int]] = None,
) -> tuple[bool, str]:
    """(ok, message). No-op when the DLLs load; otherwise download + install."""
    missing = missing_dlls(loader)
    if not missing:
        return True, ""
    log(f"\nMicrosoft Visual C++ Redistributable missing ({', '.join(missing)}): "
        "installing it — Windows will ask for permission.\n")
    exe = Path(cache_dir) / "vc_redist.x64.exe"
    try:
        (download or _download)(VC_REDIST_URL, exe, log)
    except Exception as exc:  # offline, proxy…
        return False, f"Could not download the Visual C++ Redistributable ({exc}). {MISSING_HINT}"
    try:
        code = (run_elevated or _run_elevated)(exe)
    except Exception as exc:
        return False, f"Could not start the Visual C++ Redistributable installer ({exc}). {MISSING_HINT}"
    if code == _CANCELLED:
        return False, f"Installation of the Visual C++ Redistributable was cancelled. {MISSING_HINT}"
    if code not in _OK_CODES:
        return False, f"The Visual C++ Redistributable installer exited with code {code}. {MISSING_HINT}"
    still = missing_dlls(loader)
    if still:
        return False, f"Visual C++ Redistributable installed, but {', '.join(still)} still cannot be loaded. {MISSING_HINT}"
    log("Microsoft Visual C++ Redistributable installed.\n")
    return True, "Microsoft Visual C++ Redistributable installed."
