"""Resilient model download helper."""
from __future__ import annotations

import hashlib

import pytest

from backend import model_download as md


def test_retries_with_backoff_then_succeeds(monkeypatch, tmp_path):
    target = tmp_path / "model.onnx"
    calls = {"n": 0}

    def fake_download(repo_id, filename, token=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("connection reset by peer")
        target.write_bytes(b"abc")
        return str(target)

    monkeypatch.setattr(md, "hf_hub_download", fake_download)
    path = md.download_file(
        "fake/repo", "model.onnx",
        max_attempts=3, base_delay=0,
        metadata={"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()},
    )

    assert path == str(target)
    assert calls["n"] == 3


def test_size_mismatch_is_rejected_and_cleaned_up(monkeypatch, tmp_path):
    target = tmp_path / "model.onnx"

    def fake_download(repo_id, filename, token=None, **kw):
        target.write_bytes(b"ab")  # short/corrupt file
        return str(target)

    monkeypatch.setattr(md, "hf_hub_download", fake_download)
    with pytest.raises(md.ModelDownloadError) as excinfo:
        md.download_file(
            "fake/repo", "model.onnx",
            max_attempts=2, base_delay=0, metadata={"size": 3},
        )

    err = excinfo.value
    assert err.attempts == 2
    assert "size mismatch" in str(err.cause)
    assert "Retry" in err.hint
    assert not target.exists()


def test_hash_mismatch_is_rejected(monkeypatch, tmp_path):
    target = tmp_path / "model.onnx"

    def fake_download(repo_id, filename, token=None):
        target.write_bytes(b"abc")
        return str(target)

    monkeypatch.setattr(md, "hf_hub_download", fake_download)
    with pytest.raises(md.ModelDownloadError):
        md.download_file(
            "fake/repo", "model.onnx",
            max_attempts=1, base_delay=0,
            metadata={"size": 3, "sha256": "0" * 64},
        )


def test_missing_hub_client_is_an_actionable_error(monkeypatch):
    monkeypatch.setattr(md, "hf_hub_download", None)
    with pytest.raises(md.ModelDownloadError) as excinfo:
        md.download_file("fake/repo", "model.onnx")
    assert "huggingface_hub" in str(excinfo.value)
    assert "Retry" in excinfo.value.hint


def test_corrupted_cache_blob_is_really_downloaded_again(monkeypatch, tmp_path):
    """HF cache layout: the returned path is a snapshot symlink to a blob.
    Deleting only the symlink let the next attempt relink the same corrupted
    blob; the retry must drop the blob and force a fresh download."""
    good = b"abc"
    blob = tmp_path / "blobs" / "deadbeef"
    snap = tmp_path / "snapshots" / "rev" / "model.onnx"
    blob.parent.mkdir(parents=True)
    snap.parent.mkdir(parents=True)
    calls = []

    def fake_download(repo_id, filename, token=None, force_download=False):
        calls.append(force_download)
        if force_download or not blob.exists():
            # the first transfer arrives corrupted; a cached blob is reused
            blob.write_bytes(good if force_download else b"abX")
        if not snap.is_symlink():
            snap.symlink_to(blob)               # hf relinks whatever blob is cached
        return str(snap)

    monkeypatch.setattr(md, "hf_hub_download", fake_download)
    path = md.download_file(
        "fake/repo", "model.onnx", max_attempts=2, base_delay=0,
        metadata={"size": 3, "sha256": hashlib.sha256(good).hexdigest()},
    )
    assert path == str(snap) and snap.read_bytes() == good
    assert calls == [False, True]


def test_corrupted_blob_behind_symlink_is_deleted(monkeypatch, tmp_path):
    blob = tmp_path / "blobs" / "deadbeef"
    snap = tmp_path / "snapshots" / "rev" / "model.onnx"
    blob.parent.mkdir(parents=True)
    snap.parent.mkdir(parents=True)

    def fake_download(repo_id, filename, token=None, **kw):
        blob.write_bytes(b"abX")
        if not snap.is_symlink():
            snap.symlink_to(blob)
        return str(snap)

    monkeypatch.setattr(md, "hf_hub_download", fake_download)
    with pytest.raises(md.ModelDownloadError):
        md.download_file("fake/repo", "model.onnx", max_attempts=1, base_delay=0,
                         metadata={"size": 3, "sha256": "0" * 64})
    assert not blob.exists() and not snap.is_symlink()


def test_network_error_retry_keeps_resume(monkeypatch, tmp_path):
    target = tmp_path / "model.onnx"
    calls = []

    def fake_download(repo_id, filename, token=None, **kw):
        calls.append(kw.get("force_download", False))
        if len(calls) == 1:
            raise ConnectionError("reset")
        target.write_bytes(b"abc")
        return str(target)

    monkeypatch.setattr(md, "hf_hub_download", fake_download)
    md.download_file("fake/repo", "model.onnx", max_attempts=2, base_delay=0, metadata={"size": 3})
    assert calls == [False, False]


# ── BiRefNet runs remote code: pinned revisions (I9) ────────────────────────
def test_birefnet_variants_are_pinned_to_a_commit():
    import re

    from backend.models import birefnet

    remote_code = [v for v in birefnet.VARIANTS if v != "anime"]
    assert remote_code
    for v in remote_code:
        assert re.fullmatch(r"[0-9a-f]{40}", birefnet.REVISIONS.get(v, "")), v
        assert birefnet.revision_for_repo(birefnet.VARIANTS[v]) == birefnet.REVISIONS[v]


@pytest.mark.parametrize("variant", ["general", "hr", "portrait"])
def test_birefnet_from_pretrained_gets_the_pinned_revision(monkeypatch, variant):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    transformers = pytest.importorskip("transformers")
    from backend.models import birefnet

    seen = {}

    class Stop(Exception):
        pass

    class FakeAuto:
        @staticmethod
        def from_pretrained(repo_id, **kw):
            seen.update(kw, repo_id=repo_id)
            raise Stop()

    monkeypatch.setattr(transformers, "AutoModelForImageSegmentation", FakeAuto)

    class Dev:
        torch_device = "cpu"

    m = birefnet.BiRefNetModel(Dev(), variant=variant)
    with pytest.raises(Stop):
        m._ensure_loaded()
    assert seen["repo_id"] == birefnet.VARIANTS[variant]
    assert seen["trust_remote_code"] is True
    assert seen["revision"] == birefnet.REVISIONS[variant]
    assert seen["code_revision"] == birefnet.REVISIONS[variant]


def test_birefnet_snapshot_download_is_pinned(monkeypatch):
    from backend.models import birefnet

    seen = {}

    def fake_snapshot(repo_id, token=None, max_workers=1, **kw):
        seen.update(kw, repo_id=repo_id)
        return "/cache/snap"

    class State:
        settings = {"models": {}}

    import backend.diagnostics as diag
    monkeypatch.setattr(diag, "find_model_spec",
                        lambda st, key: {"repo_id": birefnet.VARIANTS["general"], "files": None})
    monkeypatch.setattr(md, "snapshot_download", fake_snapshot)
    res = md.download_for_key(State(), "birefnet")
    assert res["ok"] and seen["revision"] == birefnet.REVISIONS["general"]
