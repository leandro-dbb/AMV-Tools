"""Status + diagnostics endpoints and log plumbing."""
from __future__ import annotations

import logging
from pathlib import Path

from backend.diagnostics import anonymize, log_file_path


def test_status_reports_cpu_fallback(client, state):
    body = client.get("/api/status").json()
    assert body["ready"] is True
    assert body["device"] == "cpu"
    assert body["requested_device"] == "auto"
    assert "CPU" in body["device_fallback"]


def test_status_reports_explicit_gpu_fallback(client, state):
    state.settings["indexing"]["device"] = "cuda"
    body = client.get("/api/status").json()
    assert body["requested_device"] == "cuda"
    assert "CUDA" in body["device_fallback"]


def test_diagnostics_contains_versions_models_disk_and_logs(client, data_dir):
    body = client.get("/api/diagnostics").json()

    assert body["app"]["version"]
    assert body["python"]
    assert body["platform"]
    assert "version" in body["ffmpeg"]
    assert {"requested", "active", "gpu_detected", "fallback_reason"} <= body["device"].keys()
    assert body["device"]["active"] == "cpu"
    assert body["device"]["gpu_detected"] is False

    assert body["data_dir"]["path"] == str(data_dir)
    assert body["data_dir"]["total_gb"] > 0
    assert body["data_dir"]["free_gb"] >= 0

    models = {m["key"]: m for m in body["models"]}
    assert set(models) == {"siglip", "wd_tagger", "birefnet", "sam2"}
    for model in models.values():
        assert {"present", "size_bytes", "loaded", "error"} <= model.keys()

    assert body["logs"]["path"].endswith("backend.log")
    assert log_file_path().exists()
    assert "AMV Tools diagnostic report" in body["markdown"]
    assert "### Models" in body["markdown"]


def test_recent_errors_are_surfaced_in_diagnostics(client):
    logging.getLogger("amv.test").error("simulated failure while loading fake/repo")
    body = client.get("/api/diagnostics").json()
    messages = [e["message"] for e in body["logs"]["recent_errors"]]
    assert any("simulated failure" in m for m in messages)


def test_markdown_report_is_anonymized(client):
    home = Path.home()
    episode = home / "Videos" / "Some Show - 04 [1080p].mkv"
    logging.getLogger("amv.test").error("cannot decode %s", episode)
    logging.getLogger("amv.test").error("cannot decode '%s'", "Other Show 12.mp4")
    md = client.get("/api/diagnostics").json()["markdown"]
    assert "cannot decode" in md
    assert str(home) not in md
    assert home.as_posix() not in md
    assert "Some Show" not in md and "Other Show" not in md
    assert "<video>.mkv" in md and "<video>.mp4" in md


def test_anonymize_handles_windows_paths_and_user_name(monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("C:\\Users\\Alice")))
    monkeypatch.setenv("USERNAME", "Alice")
    out = anonymize("log C:\\Users\\Alice\\AppData\\amv.log; D:\\Anime\\Show 01.MKV failed for alice")
    assert "Alice" not in out and "alice" not in out
    assert "Show 01" not in out
    assert "<video>.mkv" in out


def test_models_status_endpoint(client):
    body = client.get("/api/models/status").json()
    assert {m["key"] for m in body["models"]} == {"siglip", "wd_tagger", "birefnet", "sam2"}


def test_models_download_unknown_key(client):
    body = client.post("/api/models/download", json={"key": "nope"}).json()
    assert body["ok"] is False
    assert "unknown model" in body["error"]
