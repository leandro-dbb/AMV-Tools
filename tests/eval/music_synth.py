"""Synthetic music with exactly known beats, downbeats and drops.

Used by the music-sync benchmark (``tests/eval/music_bench.py``) and, in a
short form, by ``tests/test_music.py``.  Nothing here is copyrighted: every
sound is generated from sines and noise, so the ground truth is exact.

    python tests/eval/music_synth.py ~/dev/amv-testdata/music/synth
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

SR = 22_050


def _env(n: int, decay_s: float, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    return np.exp(-t / max(1e-4, decay_s)).astype(np.float32)


def _kick(sr: int = SR) -> np.ndarray:
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    freq = 45 + 110 * np.exp(-t / 0.04)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return (np.sin(phase) * _env(n, 0.12, sr)).astype(np.float32)


def _click(accent: bool, sr: int = SR) -> np.ndarray:
    n = int(0.03 * sr)
    t = np.arange(n) / sr
    f = 1760.0 if accent else 1100.0
    return (np.sin(2 * np.pi * f * t) * _env(n, 0.008, sr)).astype(np.float32)


def _noise_hit(dur: float, decay: float, rng, sr: int = SR, hp: bool = True) -> np.ndarray:
    n = int(dur * sr)
    x = rng.standard_normal(n).astype(np.float32)
    if hp:
        x = np.diff(x, prepend=0.0).astype(np.float32)
    return x * _env(n, decay, sr)


def _add(buf: np.ndarray, sound: np.ndarray, at_s: float, gain: float, sr: int = SR) -> None:
    i = int(round(at_s * sr))
    if i >= len(buf) or i < 0:
        return
    j = min(len(buf), i + len(sound))
    buf[i:j] += gain * sound[: j - i]


def _write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    y = y / max(1e-6, float(np.max(np.abs(y)))) * 0.9
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def metronome(bpm: float = 120.0, seconds: float = 40.0, meter: int = 4,
              offset: float = 0.5, sr: int = SR) -> tuple[np.ndarray, dict]:
    """Click track: accented click on each bar's first beat."""
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    period = 60.0 / bpm
    beats = np.arange(offset, seconds - 0.1, period)
    for i, t in enumerate(beats):
        _add(y, _click(i % meter == 0, sr), t, 1.0 if i % meter == 0 else 0.6, sr)
    truth = {"bpm": bpm, "beats": beats.tolist(),
             "downbeats": beats[::meter].tolist(), "drops": []}
    return y, truth


def edm(bpm: float = 128.0, sr: int = SR, seed: int = 7) -> tuple[np.ndarray, dict]:
    """Intro → build-up → drop → breakdown → build-up → drop → outro.

    Bars: intro 8, build 8, drop 16, breakdown 8, build 4, drop 16, outro 4.
    The drop starts exactly on a downbeat with kick + sub bass + clap.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = 4 * beat
    plan = [("intro", 8), ("build", 8), ("drop", 16), ("bridge", 8),
            ("build", 4), ("drop", 16), ("outro", 4)]
    offset = 0.25
    total_bars = sum(b for _, b in plan)
    seconds = offset + total_bars * bar + 2.0
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    kick = _kick(sr)
    hat = _noise_hit(0.05, 0.012, rng, sr)
    clap = _noise_hit(0.18, 0.05, rng, sr, hp=False)
    t_axis = np.arange(len(y)) / sr

    sections = []
    drops = []
    cursor = offset
    for kind, bars in plan:
        start, end = cursor, cursor + bars * bar
        sections.append({"type": kind, "start": start, "end": end})
        n_beats = bars * 4
        for k in range(n_beats):
            tb = start + k * beat
            if kind in ("intro", "outro"):
                _add(y, hat, tb + beat / 2, 0.12, sr)
                if k % 4 == 0:
                    _add(y, kick, tb, 0.25, sr)
            elif kind == "bridge":
                if k % 4 == 0:
                    _add(y, kick, tb, 0.12, sr)
            elif kind == "build":
                prog = k / n_beats
                # snare roll: 1 hit/beat, then 2, then 4 (accelerating)
                sub = 1 if prog < 0.5 else (2 if prog < 0.75 else 4)
                for s in range(sub):
                    _add(y, clap, tb + s * beat / sub, 0.15 + 0.35 * prog, sr)
                _add(y, kick, tb, 0.2 + 0.3 * prog, sr)
            elif kind == "drop":
                _add(y, kick, tb, 1.0, sr)
                _add(y, hat, tb + beat / 2, 0.3, sr)
                if k % 2 == 1:
                    _add(y, clap, tb, 0.6, sr)
        mask = (t_axis >= start) & (t_axis < end)
        tt = t_axis[mask] - start
        if kind in ("intro", "bridge", "outro"):
            pad = 0.10 * (np.sin(2 * np.pi * 220 * tt) + 0.6 * np.sin(2 * np.pi * 330 * tt))
            y[mask] += pad.astype(np.float32)
        elif kind == "build":
            prog = tt / (end - start)
            riser = rng.standard_normal(tt.size).astype(np.float32)
            riser = np.diff(riser, prepend=0.0) * (0.02 + 0.2 * prog ** 2)
            y[mask] += riser.astype(np.float32)
        elif kind == "drop":
            drops.append(start)
            # Sub bass on every beat (55 Hz), ducked by the kick.
            phase = (tt % beat) / beat
            bass = 0.5 * np.sin(2 * np.pi * 55 * tt) * np.clip(phase * 4, 0, 1)
            y[mask] += bass.astype(np.float32)
        cursor = end

    beats = offset + beat * np.arange(total_bars * 4)
    truth = {"bpm": bpm, "beats": beats.tolist(), "downbeats": beats[::4].tolist(),
             "drops": drops, "sections": sections}
    return y, truth


def tempo_ramp(bpm_from: float = 90.0, bpm_to: float = 130.0, seconds: float = 60.0,
               sr: int = SR, seed: int = 3) -> tuple[np.ndarray, dict]:
    """Kick + hats whose tempo glides linearly (variable-tempo case)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    kick = _kick(sr)
    hat = _noise_hit(0.05, 0.012, rng, sr)
    beats = []
    t = 0.3
    i = 0
    while t < seconds - 0.5:
        beats.append(t)
        _add(y, kick, t, 1.0 if i % 4 == 0 else 0.7, sr)
        bpm = bpm_from + (bpm_to - bpm_from) * (t / seconds)
        period = 60.0 / bpm
        _add(y, hat, t + period / 2, 0.3, sr)
        t += period
        i += 1
    truth = {"bpm": (bpm_from + bpm_to) / 2, "beats": beats, "downbeats": beats[::4],
             "drops": []}
    return y, truth


def slow(bpm: float = 68.0, seconds: float = 50.0, sr: int = SR) -> tuple[np.ndarray, dict]:
    """Soft piano-like notes on each beat (no drums): the hard, calm case."""
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    period = 60.0 / bpm
    beats = np.arange(0.4, seconds - 1.0, period)
    notes = [261.6, 329.6, 392.0, 329.6, 293.7, 349.2, 440.0, 349.2]
    for i, t in enumerate(beats):
        n = int(1.6 * sr)
        tt = np.arange(n) / sr
        f = notes[i % len(notes)]
        tone = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt))
        att = np.clip(tt / 0.01, 0, 1)
        _add(y, (tone * att * _env(n, 0.5, sr)).astype(np.float32), t,
             0.9 if i % 4 == 0 else 0.6, sr)
    truth = {"bpm": bpm, "beats": beats.tolist(), "downbeats": beats[::4].tolist(),
             "drops": []}
    return y, truth


TRACKS = {
    "click120": lambda: metronome(120.0, 40.0),
    "edm128": edm,
    "ramp90-130": tempo_ramp,
    "slow68": slow,
}


def main(out_dir: str) -> None:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for name, make in TRACKS.items():
        y, truth = make()
        _write_wav(out / f"{name}.wav", y)
        (out / f"{name}.json").write_text(json.dumps(truth, indent=1))
        print(f"{name}: {len(y) / SR:.1f}s, {len(truth['beats'])} beats, drops={truth['drops']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/dev/amv-testdata/music/synth")
