"""First-indexing model prefetch (backend/model_prefetch.py): visible
progress while SigLIP/tagger download, never a crash, never the network for
injected test models."""
import time

from backend import model_prefetch as mp


class _State:
    pass


def _specs(_state):
    return [
        {"key": "siglip", "label": "SigLIP 2", "repo_id": "google/siglip2-base-patch16-naflex", "files": None},
        {"key": "wd_tagger", "label": "wd-tagger", "repo_id": "SmilingWolf/wd-vit-tagger-v3", "files": ["model.onnx"]},
        {"key": "birefnet", "label": "x", "repo_id": "x/y", "files": None},
    ]


def test_progress_event_text_and_percent():
    e = mp.progress_event("SigLIP 2 base (NaFlex)", 512 * 2**20, 1500)
    assert e["type"] == "indexing" and e["percent"] == 34
    assert e["video"] == "SigLIP 2 base (NaFlex) ↓ 512 / 1500 MB"
    assert mp.progress_event("x", 2**20, None)["percent"] == 0
    assert mp.progress_event("x", 10**12, 10)["percent"] == 99   # 100 only once verified


def test_prefetch_downloads_missing_models_with_progress(monkeypatch, tmp_path):
    import backend.diagnostics as diag
    import backend.model_download as md
    monkeypatch.setattr(mp, "_real_wrappers", lambda s: {"siglip", "wd_tagger"})
    monkeypatch.setattr(diag, "_model_specs", _specs)
    monkeypatch.setattr(diag, "_hf_cache_index", lambda: {"SmilingWolf/wd-vit-tagger-v3": {"size_bytes": 380 * 2**20}})
    monkeypatch.setattr(mp, "_repo_cache_dir", lambda repo: tmp_path / repo.replace("/", "--"))
    calls = []

    def fake_download(state, key):
        calls.append(key)
        blobs = tmp_path / "google--siglip2-base-patch16-naflex" / "blobs"
        blobs.mkdir(parents=True)
        for i in range(3):
            (blobs / "w.incomplete").write_bytes(b"x" * (i + 1) * 2**20)
            time.sleep(0.05)
        return {"ok": True}

    monkeypatch.setattr(md, "download_for_key", fake_download)
    events = []
    fetched = mp.prefetch_indexing_models(_State(), events.append, interval_s=0.01)
    assert calls == ["siglip"]            # tagger already cached, birefnet not an indexing model
    assert fetched == ["google/siglip2-base-patch16-naflex"]
    assert events[0]["video"].startswith("SigLIP 2 base (NaFlex) ↓ 0 / 1500 MB")
    assert events[-1]["percent"] == 100
    assert all(e["type"] == "indexing" for e in events)


def test_prefetch_failure_is_logged_not_raised(monkeypatch, tmp_path):
    import backend.diagnostics as diag
    import backend.model_download as md
    monkeypatch.setattr(mp, "_real_wrappers", lambda s: {"siglip"})
    monkeypatch.setattr(diag, "_model_specs", _specs)
    monkeypatch.setattr(diag, "_hf_cache_index", lambda: {})
    monkeypatch.setattr(mp, "_repo_cache_dir", lambda repo: tmp_path)
    monkeypatch.setattr(md, "download_for_key", lambda s, k: {"ok": False, "error": "offline", "hint": "Check your internet"})
    assert mp.prefetch_indexing_models(_State(), lambda e: None, interval_s=0.01) == []

    def boom(s, k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(md, "download_for_key", boom)
    assert mp.prefetch_indexing_models(_State(), lambda e: None, interval_s=0.01) == []


def test_injected_fake_models_never_reach_the_network(monkeypatch):
    import backend.model_download as md
    state = _State()
    state._siglip = object()
    state._tagger = object()
    monkeypatch.setattr(md, "download_for_key", lambda *a: (_ for _ in ()).throw(AssertionError("network!")))
    assert mp.prefetch_indexing_models(state, lambda e: None) == []
