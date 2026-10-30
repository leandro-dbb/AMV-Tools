"""Beat This! (vendored, optional): front end equal to torchaudio's (checked
against the real package on debian-nas: mel max diff 5e-5, identical beats
and first beats on a real song), graceful absence, bar-phase choice."""
from __future__ import annotations

import numpy as np
import pytest

from backend.music import analysis, beatthis


def test_mel_filterbank_matches_librosa_slaney():
    librosa = pytest.importorskip("librosa")
    from backend.music.beatthis.inference import mel_filterbank
    ref = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=128, fmin=30.0, fmax=11000.0, htk=False, norm=None)
    assert np.allclose(mel_filterbank(), ref.T, atol=1e-4)


def test_model_absent_means_heuristic(monkeypatch):
    monkeypatch.setattr(beatthis, "checkpoint", lambda: None)
    assert beatthis.beats_and_downbeats(np.zeros(22050, np.float32), 22050) is None
    assert analysis._beat_this_phase(np.zeros(22050, np.float32), 22050, np.arange(0, 10, 0.5)) is None


def test_registered_for_download():
    from backend.finish import registry
    spec = registry.MODELS[beatthis.MODEL_ID]
    assert spec.license == "MIT" and spec.files[0].sha256 and spec.size < 10_000_000


def test_phase_follows_its_first_beats(monkeypatch):
    beats = np.arange(0.0, 20.0, 0.5)
    downs = beats[2::4]
    monkeypatch.setattr(beatthis, "beats_and_downbeats", lambda y, sr: (beats, downs + 0.02))
    assert analysis._beat_this_phase(np.zeros(10), 22050, beats) == 2
    # another tempo octave / grid: no agreement, no override
    monkeypatch.setattr(beatthis, "beats_and_downbeats", lambda y, sr: (beats, downs + 0.25))
    assert analysis._beat_this_phase(np.zeros(10), 22050, beats) is None
