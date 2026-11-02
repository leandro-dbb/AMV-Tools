"""Windows-only first-run fixes found on a real Windows 11 + RTX 3090 VM.

- A fresh Windows has no Visual C++ Redistributable: torch cannot load
  (WinError 126 on c10.dll). The setup installs it (backend.winruntime).
- The backend validation decoded the child's UTF-8 output as cp1252, so a
  French Windows error read "Le module spÃ©cifiÃ© est introuvable".
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from backend import winruntime
from backend.api import setup as setup_api


def _loader(missing: set[str]):
    def load(name: str):
        if name in missing:
            raise OSError(126, "module not found")
        return object()
    return load


def test_nothing_to_do_when_the_runtime_loads(tmp_path):
    calls = []
    ok, msg = winruntime.ensure_msvc_runtime(
        tmp_path, calls.append, loader=_loader(set()),
        download=lambda *a: calls.append("download"), run_elevated=lambda exe: calls.append("run"))
    assert ok and msg == "" and calls == []


def test_missing_runtime_is_downloaded_and_installed(tmp_path):
    missing = {"msvcp140.dll", "vcruntime140_1.dll"}
    seen = {}

    def download(url, dest, log):
        seen["url"], seen["dest"] = url, dest

    def run(exe):
        seen["exe"] = exe
        missing.clear()          # the redistributable put the DLLs in System32
        return 0

    lines = []
    ok, msg = winruntime.ensure_msvc_runtime(tmp_path, lines.append, loader=_loader(missing),
                                             download=download, run_elevated=run)
    assert ok, msg
    assert seen["url"] == winruntime.VC_REDIST_URL
    assert seen["exe"] == seen["dest"] == tmp_path / "vc_redist.x64.exe"
    assert "msvcp140.dll" in lines[0] and "permission" in lines[0]


@pytest.mark.parametrize("code", [1638, 3010])
def test_already_newer_or_reboot_pending_counts_as_installed(tmp_path, code):
    missing = {"msvcp140.dll"}

    def run(exe):
        missing.clear()
        return code
    ok, _ = winruntime.ensure_msvc_runtime(tmp_path, lambda s: None, loader=_loader(missing),
                                           download=lambda *a: None, run_elevated=run)
    assert ok


def test_refused_permission_prompt_gives_the_manual_link(tmp_path):
    ok, msg = winruntime.ensure_msvc_runtime(
        tmp_path, lambda s: None, loader=_loader({"msvcp140.dll"}),
        download=lambda *a: None, run_elevated=lambda exe: 1223)
    assert not ok
    assert "cancelled" in msg and winruntime.VC_REDIST_URL in msg


def test_offline_download_failure_is_explained(tmp_path):
    def download(*a):
        raise OSError("getaddrinfo failed")
    ok, msg = winruntime.ensure_msvc_runtime(tmp_path, lambda s: None, loader=_loader({"msvcp140.dll"}),
                                             download=download, run_elevated=lambda exe: 0)
    assert not ok and "getaddrinfo failed" in msg and winruntime.VC_REDIST_URL in msg


def test_installer_that_does_not_fix_it_is_reported(tmp_path):
    ok, msg = winruntime.ensure_msvc_runtime(
        tmp_path, lambda s: None, loader=_loader({"msvcp140.dll"}),
        download=lambda *a: None, run_elevated=lambda exe: 0)
    assert not ok and "msvcp140.dll still cannot be loaded" in msg


def test_not_windows_means_nothing_missing():
    if sys.platform == "win32":
        pytest.skip("real check on Windows")
    assert winruntime.missing_dlls() == []


def test_torch_dll_error_is_recognised():
    log = ('Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.\n'
           'OSError: [WinError 126] Le module spécifié est introuvable. Error loading "…\\torch\\lib\\c10.dll"')
    assert winruntime.looks_like_missing_runtime(log)
    assert not winruntime.looks_like_missing_runtime("torch 2.11.0 has CUDA 13.0, but no CUDA device is available")


def test_validation_output_is_decoded_as_utf8(monkeypatch):
    captured = {}

    def fake_run(cmd, **kw):
        captured.update(kw)
        out = "Le module spécifié est introuvable".encode("utf-8")
        text = out.decode(kw.get("encoding") or "cp1252", errors=kw.get("errors", "strict"))
        return subprocess.CompletedProcess(cmd, 1, stdout=text, stderr="")

    monkeypatch.setattr(setup_api.subprocess, "run", fake_run)
    ok, log = setup_api._validate_backend("cu130")
    assert not ok
    assert log == "Le module spécifié est introuvable"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"


def test_elevated_command_carries_the_installer_path(monkeypatch, tmp_path):
    # Regression (real VM): the path was passed after -Command, where it never
    # reaches $args -> Start-Process without a file, exit code 1.
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(winruntime.subprocess, "run", fake_run)
    exe = tmp_path / "Données d'Élise" / "vc_redist.x64.exe"
    assert winruntime._run_elevated(exe) == 0
    script = seen["cmd"][-1]
    assert seen["cmd"][-2] == "-Command"
    assert "-FilePath '" + str(exe).replace("'", "''") + "'" in script
    assert "-Verb RunAs" in script and "/install" in script


def test_refused_prompt_maps_to_cancelled(monkeypatch, tmp_path):
    monkeypatch.setattr(winruntime.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, stdout="", stderr="Start-Process : Cette commande ne peut pas être exécutée … L'opération a été annulée par l'utilisateur."))
    assert winruntime._run_elevated(tmp_path / "x.exe") == winruntime._CANCELLED


def test_autotune_waits_for_the_first_run_setup(tmp_path, monkeypatch):
    # Real VM: the autotune imported torch 20 s after launch, while the
    # onboarding screen was still up (VC++ runtime missing): a half-imported
    # torch stayed in the sidecar -> "GPU unavailable" banner after setup.
    from backend import autotune
    monkeypatch.setattr("backend.paths.user_data_dir", lambda: tmp_path)
    ran = []
    monkeypatch.setattr(autotune, "run", lambda **kw: ran.append(1))
    monkeypatch.setattr(autotune, "needs_run", lambda entry: True)
    monkeypatch.setattr(autotune, "_busy", lambda: False)
    autotune._background(False, 0)
    assert ran == [] and autotune._status["state"] == "waiting_setup"
    (tmp_path / ".setup_complete").write_text("cu130")
    autotune._background(False, 0)
    assert ran == [1]


def test_successful_setup_starts_the_autotune(client, data_dir, monkeypatch):
    from backend import autotune
    started = []
    monkeypatch.setattr(autotune, "start_background", lambda *a, **k: started.append(1) or True)
    monkeypatch.setenv("AMV_E2E_SKIP_SYNC", "1")
    (data_dir / ".setup_complete").unlink(missing_ok=True)
    r = client.post("/api/setup/install", json={"backend": "cpu"}).json()
    assert r["ok"] and started == [1]
