"""Tempo, beats, downbeats, onsets, energy curve, sections and drops.

Design notes (the measured "why" is in docs/reports/v06-C-synchro-musique.md):

* One STFT feeds everything (onset envelope, RMS, bass band, chroma, MFCC):
  a 4-minute song is analysed in a few seconds on one CPU core, no model
  weights, so the feature works on every machine.
* Beats: librosa's dynamic-programming tracker (ISC licence) on a mel
  spectral-flux envelope.  When the local tempo drifts (live drummer, tempo
  ramps) the tracker is fed the time-varying tempo instead of one global BPM.
* Downbeats: the beat phase (0-3 in 4/4) where harmony changes, bass and
  attacks concentrate — bars start where chords and sections change.
* Sections: self-similarity novelty on bar-level timbre/harmony/energy
  features (Foote checkerboard kernel) plus energy jumps, so boundaries fall
  on bar lines.  Each section is then labelled from its energy relative to
  the whole song.
* Drops: a section that starts with a strong jump of loudness *and* bass
  after a quieter stretch; the bars that rise just before it become the
  build-up.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .audio import DEFAULT_SR, decode_audio, fingerprint_file

ANALYSIS_VERSION = 2
HOP = 512
N_FFT = 2048
METER = 4
LOW_WEIGHT = 4.0
# The flux of a centred STFT frame peaks about a quarter window before the
# attack (measured -20…-28 ms on click / kick tracks): shift beats back.
LATENCY_S = N_FFT / 4 / DEFAULT_SR
SECTION_TYPES = ("intro", "verse", "build", "drop", "chorus", "bridge", "outro")

# Drop = energy jump into a loud section, with the bass coming in too.
_DROP_JUMP = 0.17
_DROP_BASS_JUMP = 0.12
# Minimum section length in bars (shorter novelty peaks are merged).
_MIN_SECTION_BARS = 4


def _ms(t: float) -> int:
    return int(round(float(t) * 1000.0))


def _norm01(values: np.ndarray, lo_pct: float = 5, hi_pct: float = 98) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _smooth(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(np.asarray(values, dtype=np.float32),
                             sigma=max(0.5, sigma_frames), mode="nearest")


def _span_mean(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Column i = mean of x[:, bounds[i]:bounds[i+1]] (frames)."""
    x = np.atleast_2d(x)
    out = np.zeros((x.shape[0], max(0, len(bounds) - 1)), dtype=np.float32)
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        a = int(a)
        b = max(a + 1, int(b))
        out[:, i] = x[:, a:b].mean(axis=1) if a < x.shape[1] else x[:, -1]
    return out


# ── low-level features ──────────────────────────────────────────────────────
def _features(y: np.ndarray, sr: int) -> dict:
    import librosa

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)).astype(np.float32)
    power = S ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Spectral flux per mel band, weighted towards the low end: kicks and
    # snares carry the beat, while hi-hats sit on the off-beats and would
    # otherwise pull the tracker half a beat early.
    flux = np.maximum(0.0, np.diff(mel_db, axis=1, prepend=mel_db[:, :1]))
    weights = np.geomspace(LOW_WEIGHT, 1.0 / LOW_WEIGHT, mel_db.shape[0])[:, None]
    onset_env = np.mean(flux * weights, axis=0)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    low_bins = freqs <= 150.0
    rms = np.sqrt(np.mean(power, axis=0))
    low = np.sqrt(np.mean(power[low_bins], axis=0))
    # Loudness is perceived in dB: normalise the dB curves, not the linear RMS.
    rms_db = 20.0 * np.log10(rms + 1e-6)
    low_db = 20.0 * np.log10(low + 1e-6)
    # tuning=0: estimating the tuning (piptrack) costs more than the whole
    # beat tracker and only shifts chroma bins by a fraction of a semitone.
    chroma = librosa.feature.chroma_stft(S=power, sr=sr, hop_length=HOP, tuning=0.0)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=13)
    n = min(len(onset_env), S.shape[1])
    return {
        "onset_env": onset_env[:n],
        "loud": _norm01(rms_db[:n]),
        "loud_db": rms_db[:n].astype(np.float32),
        "bass": _norm01(low_db[:n]),
        "chroma": chroma[:, :n],
        "mfcc": mfcc[:, :n],
        "n": n,
    }


def _track_beats(onset_env: np.ndarray, sr: int, duration: float) -> tuple[float, np.ndarray, bool]:
    """(bpm, beat times, variable_tempo)."""
    import librosa

    tempo = float(np.atleast_1d(librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=HOP))[0])
    # AMV songs live between ~60 and ~180 BPM; outside that the estimate is
    # almost always an octave error (hi-hats counted as beats, or half-time).
    while tempo > 180.0:
        tempo /= 2.0
    while 0 < tempo < 60.0:
        tempo *= 2.0
    local = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=HOP,
                                  aggregate=None)
    local = np.asarray(local, dtype=np.float64)
    variable = False
    bpm_arg: float | np.ndarray = tempo
    if local.size > 8 and np.all(np.isfinite(local)):
        # Ignore octave flips of the local estimate (60 ↔ 120 BPM) when
        # judging drift: only a gradual change means a real tempo change.
        folded = local * np.where(local < tempo * 0.75, 2.0, 1.0)
        folded = folded * np.where(folded > tempo * 1.5, 0.5, 1.0)
        folded = _smooth(folded, 40).astype(np.float64)
        p10, p90 = np.percentile(folded, [10, 90])
        if p90 / max(1e-6, p10) > 1.12:
            variable = True
            bpm_arg = folded
    _, frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr,
                                        hop_length=HOP, bpm=bpm_arg, trim=True,
                                        units="frames")
    beats = librosa.frames_to_time(np.asarray(frames, dtype=int), sr=sr, hop_length=HOP)
    beats = beats + LATENCY_S
    beats = np.asarray([t for t in beats if 0.0 <= t < duration], dtype=np.float64)
    if beats.size >= 4:
        # trim=True drops weak beats in silent heads/tails; re-extend the
        # grid at the local period so every second of the song can be cut on.
        period = float(np.median(np.diff(beats[:8])))
        head = np.arange(beats[0] - period, -period * 0.25, -period)[::-1]
        head = head[head >= 0.0]
        period_end = float(np.median(np.diff(beats[-8:])))
        tail = np.arange(beats[-1] + period_end, duration - period_end * 0.25, period_end)
        beats = np.concatenate([head, beats, tail])
        intervals = np.diff(beats)
        med = float(np.median(intervals))
        steady = intervals[(intervals > 0.6 * med) & (intervals < 1.6 * med)]
        # mean, not median: beat frames are quantised to 23 ms hops, the
        # median interval snaps to one of two hop counts.
        tempo = float(60.0 / np.mean(steady if steady.size else intervals))
    else:
        # Too sparse to track (ambient intro, silence): constant grid.
        period = 60.0 / max(40.0, tempo or 120.0)
        beats = np.arange(0.0, duration, period)
    return tempo, beats, variable


def _downbeat_phase(beat_frames: np.ndarray, feats: dict) -> int:
    """Beat phase (0..METER-1) where bars most likely start."""
    import librosa

    nb = len(beat_frames)
    if nb < METER * 2:
        return 0
    n = feats["n"]
    bounds = np.clip(beat_frames, 0, n - 1)
    chroma = _span_mean(feats["chroma"], np.append(bounds, n))
    cnorm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    change = np.zeros(nb)
    change[1:] = 1.0 - np.sum(cnorm[:, 1:] * cnorm[:, :-1], axis=0)
    attack = feats["onset_env"][bounds]
    bass = feats["bass"][bounds]
    # accented beat = louder right on the beat than its neighbours
    loud = np.asarray([float(np.max(feats["loud_db"][f:f + 4])) for f in bounds])

    def z(v: np.ndarray) -> np.ndarray:
        return (v - np.mean(v)) / (np.std(v) + 1e-9)

    strength = z(change) + 0.7 * z(attack) + 0.5 * z(bass) + 0.7 * z(loud)
    scores = [float(np.mean(strength[p::METER])) for p in range(METER)]
    return int(np.argmax(scores))


# ── sections ────────────────────────────────────────────────────────────────
def _bar_features(bar_frames: np.ndarray, feats: dict, intensity: np.ndarray) -> dict:
    import librosa

    n = feats["n"]
    nbars = len(bar_frames) - 1
    idx = np.clip(bar_frames, 0, n - 1)

    def sync(x: np.ndarray) -> np.ndarray:
        return _span_mean(x, idx)

    return {
        "intensity": sync(intensity)[0],
        "bass": sync(feats["bass"])[0],
        "loud": sync(feats["loud"])[0],
        "timbre": sync(feats["mfcc"]),
        "harmony": sync(feats["chroma"]),
        "attack": sync(feats["onset_env"])[0],
    }


def _novelty(bars: dict, width: int = 4) -> np.ndarray:
    """Foote novelty on a bar-level self-similarity matrix."""
    def zs(m: np.ndarray) -> np.ndarray:
        m = np.atleast_2d(m)
        return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)

    X = np.vstack([
        zs(bars["timbre"][1:]),               # drop MFCC 0 (≈ loudness, counted below)
        0.6 * zs(bars["harmony"]),
        2.5 * zs(bars["intensity"]),
        1.5 * zs(bars["bass"]),
    ])
    n = X.shape[1]
    if n < 2 * width + 1:
        return np.zeros(n)
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    ssm = Xn.T @ Xn
    g = np.exp(-0.5 * (np.linspace(-1.5, 1.5, 2 * width)) ** 2)
    taper = np.outer(g, g)
    sign = np.ones((2 * width, 2 * width))
    sign[:width, width:] = -1
    sign[width:, :width] = -1
    kernel = taper * sign
    pad = np.pad(ssm, width, mode="edge")
    nov = np.zeros(n)
    for i in range(n):
        # boundary *before* bar i: window centred between bars i-1 and i
        block = pad[i:i + 2 * width, i:i + 2 * width]
        nov[i] = float(np.sum(block * kernel))
    nov[0] = 0.0
    return np.maximum(nov, 0.0)


def _pick_boundaries(bars: dict, nov: np.ndarray) -> list[int]:
    """Bar indices where a new section starts (always includes 0)."""
    nbars = len(nov)
    inten = bars["intensity"]
    cands: list[tuple[float, int]] = []
    if nbars:
        peak = float(np.max(nov)) + 1e-9
        thr = float(np.mean(nov) + 0.3 * np.std(nov))
        for i in range(1, nbars):
            left = nov[i - 1]
            right = nov[i + 1] if i + 1 < nbars else -1
            if nov[i] >= thr and nov[i] >= left and nov[i] >= right:
                # Energy steps weigh extra: a drop or a breakdown is exactly
                # "same loop, much louder / quieter".
                step = abs(float(np.mean(inten[i:i + 2])) - float(np.mean(inten[max(0, i - 2):i])))
                cands.append((float(nov[i]) / peak + 2.0 * step, i))
    cands.sort(reverse=True)
    chosen: list[int] = [0]
    for _score, i in cands:
        if i < _MIN_SECTION_BARS // 2 or nbars - i < 2:
            continue
        if any(abs(i - c) < _MIN_SECTION_BARS for c in chosen):
            continue
        chosen.append(i)
    return sorted(chosen)


def _label(segments: list[dict], bars: dict) -> None:
    """Assign a type to every segment (in place)."""
    inten = bars["intensity"]
    bass = bars["bass"]
    nseg = len(segments)
    if not nseg:
        return
    hi = max(0.5, float(np.percentile(inten, 65)))
    lo = min(0.45, float(np.percentile(inten, 30)))
    for s in segments:
        a, b = s["bar_start"], s["bar_end"]
        s["energy"] = float(np.mean(inten[a:b]))
        s["bass_level"] = float(np.mean(bass[a:b]))
        s["head"] = float(np.mean(inten[a:min(b, a + 2)]))
        s["bass_head"] = float(np.mean(bass[a:min(b, a + 2)]))
        half = max(1, (b - a) // 2)
        s["slope"] = float(np.mean(inten[a + half:b]) - np.mean(inten[a:a + half])) if b - a >= 2 else 0.0

    for i, s in enumerate(segments):
        if i > 0:
            # Compared with the whole previous section: a build-up ends loud,
            # but on average it sits well below the drop it leads to.
            p = segments[i - 1]
            s["jump"] = s["head"] - p["energy"]
            s["bass_jump"] = s["bass_head"] - p["bass_level"]
        else:
            s["jump"] = s["bass_jump"] = 0.0
        loud = s["energy"] >= hi
        if i > 0 and loud and s["jump"] >= _DROP_JUMP and s["bass_jump"] >= _DROP_BASS_JUMP:
            s["type"] = "drop"
        elif loud:
            s["type"] = "chorus"
        elif i == 0:
            s["type"] = "intro"
        elif i == nseg - 1:
            s["type"] = "outro"
        elif s["energy"] <= lo:
            s["type"] = "bridge"
        else:
            s["type"] = "verse"
    # A song that starts loud has no intro; one that ends loud has no outro.
    if segments[0]["type"] in ("verse", "bridge"):
        segments[0]["type"] = "intro"
    if nseg > 1 and segments[-1]["type"] in ("verse", "bridge") \
            and segments[-1]["slope"] <= 0.05:
        segments[-1]["type"] = "outro"
    # Build-ups: what comes right before a drop (or a loud chorus) and rises.
    for i in range(1, nseg):
        if segments[i]["type"] not in ("drop", "chorus"):
            continue
        p = segments[i - 1]
        if p["type"] in ("drop", "chorus") or i - 1 == 0 and p["type"] == "intro" and p["slope"] < 0.08:
            continue
        if p["slope"] > 0.04 or segments[i]["type"] == "drop":
            p["type"] = "build"


def _split_build(segments: list[dict], bars: dict) -> list[dict]:
    """A long section before a drop: its rising tail becomes the build-up."""
    inten = bars["intensity"]
    out: list[dict] = []
    for i, s in enumerate(segments):
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        a, b = s["bar_start"], s["bar_end"]
        if nxt is not None and nxt["type"] == "drop" and s["type"] != "build" and b - a >= 12:
            tail = inten[b - 8:b]
            start = b - 8 + int(np.argmin(tail))
            start = max(a + 4, start - (start - a) % 2)
            if b - start >= 2 and float(inten[b - 1] - inten[start]) > 0.06:
                head = dict(s, bar_end=start)
                build = dict(s, bar_start=start, type="build")
                out.extend([head, build])
                continue
        if nxt is not None and nxt["type"] == "drop" and s["type"] not in ("build", "intro") \
                and b - a < 12:
            s = dict(s, type="build")
        out.append(s)
    return out


def _merge_same(segments: list[dict]) -> list[dict]:
    """Neighbours with the same label read as one section (a drop followed
    by an equally loud stretch is still the drop)."""
    out: list[dict] = []
    for s in segments:
        if out:
            p = out[-1]
            same = p["type"] == s["type"] and s["type"] != "drop"
            drop_tail = p["type"] == "drop" and s["type"] == "chorus"
            if same or drop_tail:
                p["bar_end"] = s["bar_end"]
                continue
        out.append(dict(s))
    return out


def _snap(t: float, candidates: np.ndarray, tol: float) -> float:
    if candidates.size:
        i = int(np.argmin(np.abs(candidates - t)))
        if abs(candidates[i] - t) <= tol:
            return float(candidates[i])
    return float(t)


def _refine_drop(start: float, bar_s: float, intensity: np.ndarray, times: np.ndarray,
                 beats: np.ndarray, strong: np.ndarray) -> float:
    """Exact drop time.

    The section boundary sits on a bar line; when the bar phase is wrong it
    can trail the real drop by a beat or two.  If a much sharper loudness +
    bass rise exists within a bar of it, the drop is there (snapped to a
    beat or a strong attack when one is within 0.1 s); otherwise the
    boundary is kept (a drop can be a texture change at equal loudness).
    """
    if times.size < 4:
        return start
    frame_s = float(times[1] - times[0])
    half = max(2, int(round(0.8 / frame_s)))

    def rise(i: int) -> float:
        return float(np.mean(intensity[i:i + half]) - np.mean(intensity[i - half:i]))

    lo = max(half, int(np.searchsorted(times, start - bar_s)))
    hi = min(len(intensity) - half, int(np.searchsorted(times, start + 0.5 * bar_s)))
    i0 = int(np.clip(np.searchsorted(times, start), half, max(half, len(intensity) - half - 1)))
    t = start
    if hi > lo:
        rises = [rise(i) for i in range(lo, hi)]
        k = int(np.argmax(rises))
        if rises[k] >= 0.25 and rises[k] > rise(i0) + 0.1:
            t = float(times[lo + k])
    snapped = _snap(t, strong, 0.07)
    if snapped == t:
        snapped = _snap(t, beats, 0.1)
    return snapped


def _segments(y: np.ndarray, sr: int) -> dict:
    import librosa

    duration = float(len(y)) / sr
    feats = _features(y, sr)
    n = feats["n"]
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP)
    intensity = _smooth(0.6 * feats["loud"] + 0.4 * feats["bass"], 0.15 * sr / HOP)

    tempo, beats, variable = _track_beats(feats["onset_env"], sr, duration)
    beat_frames = np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=HOP), 0, n - 1)
    phase = _downbeat_phase(beat_frames, feats)
    downbeats = beats[phase::METER]

    onset_frames = librosa.onset.onset_detect(onset_envelope=feats["onset_env"], sr=sr,
                                              hop_length=HOP, units="frames")
    onset_frames = np.clip(np.asarray(onset_frames, dtype=int), 0, n - 1)
    onsets = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP) + LATENCY_S
    strengths = feats["onset_env"][onset_frames] if onset_frames.size else np.zeros(0)
    if strengths.size:
        strong = onsets[strengths >= np.percentile(strengths, 85)]
    else:
        strong = np.zeros(0)

    # Bars: from each downbeat to the next (plus the pickup before the first).
    bar_times = list(downbeats)
    if not bar_times or bar_times[0] > 0.05:
        bar_times.insert(0, 0.0)
    bar_times.append(duration)
    bar_times = np.asarray(bar_times)
    bar_frames = np.clip(librosa.time_to_frames(bar_times, sr=sr, hop_length=HOP), 0, n)

    sections: list[dict] = []
    if len(bar_times) >= 3:
        bars = _bar_features(bar_frames, feats, intensity)
        nbars = len(bar_times) - 1
        bounds = _pick_boundaries(bars, _novelty(bars))
        segs = [{"bar_start": a, "bar_end": b}
                for a, b in zip(bounds, bounds[1:] + [nbars]) if b > a]
        _label(segs, bars)
        segs = _merge_same(segs)
        segs = _split_build(segs, bars)
        _label_energy = bars["intensity"]
        for s in segs:
            a, b = s["bar_start"], s["bar_end"]
            start = float(bar_times[a])
            end = float(bar_times[b]) if b < len(bar_times) else duration
            if s["type"] == "drop":
                start = _refine_drop(start, float(np.median(np.diff(bar_times))),
                                     intensity, times, beats, strong)
            sections.append({
                "type": s["type"],
                "start": start,
                "end": end,
                "energy": float(np.mean(_label_energy[a:b])) if b > a else 0.0,
            })
        # keep contiguity after snapping
        for prev, cur in zip(sections, sections[1:]):
            prev["end"] = cur["start"]
        if sections:
            sections[0]["start"] = 0.0
            sections[-1]["end"] = duration
    else:
        sections.append({"type": "verse", "start": 0.0, "end": duration,
                         "energy": float(np.mean(intensity)) if n else 0.0})

    drops = [s["start"] for s in sections if s["type"] == "drop"]

    # ── UI-friendly curves ─────────────────────────────────────────────────
    step_s = 0.1
    grid = np.arange(0.0, duration, step_s)
    # The shown curve (and the shot-matching target) is smoothed over ~half
    # a second: per-beat pumping is rhythm, not the passage's intensity.
    calm_curve = _smooth(intensity, 0.5 * sr / HOP)
    energy_grid = np.interp(grid, times, calm_curve).clip(0.0, 1.0)
    buckets = 1500
    edges = np.linspace(0, len(y), buckets + 1).astype(int)
    peak = float(np.max(np.abs(y))) or 1.0
    waveform = [round(float(np.max(np.abs(y[a:b]))) / peak, 3) if b > a else 0.0
                for a, b in zip(edges[:-1], edges[1:])]

    return {
        "duration_ms": _ms(duration),
        "bpm": round(float(tempo), 2),
        "variable_tempo": bool(variable),
        "meter": METER,
        "beats_ms": [_ms(t) for t in beats],
        "downbeats_ms": [_ms(t) for t in downbeats],
        "onsets_ms": [_ms(t) for t in onsets],
        "strong_onsets_ms": [_ms(t) for t in strong],
        "drops_ms": [_ms(t) for t in drops],
        "sections": [
            {"index": i, "type": s["type"], "start_ms": _ms(s["start"]),
             "end_ms": _ms(s["end"]), "energy": round(s["energy"], 3),
             "drop": s["type"] == "drop"}
            for i, s in enumerate(sections)
        ],
        "energy": {"step_ms": int(step_s * 1000),
                   "values": [round(float(v), 3) for v in energy_grid]},
        "waveform": waveform,
    }


def analyze_signal(y: np.ndarray, sr: int = DEFAULT_SR) -> dict:
    """Analyse an in-memory mono signal (the tests feed synthetic tracks)."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    t0 = time.perf_counter()
    data = _segments(y, int(sr))
    data["analysis_version"] = ANALYSIS_VERSION
    data["analysis_seconds"] = round(time.perf_counter() - t0, 3)
    return data


def analyze_audio(path: str, *, sr: int = DEFAULT_SR, file_hash: Optional[str] = None) -> dict:
    """Decode + fingerprint + analyse a file.  The caller handles caching."""
    t0 = time.perf_counter()
    y = decode_audio(path, sr=sr)
    data = analyze_signal(y, sr)
    data.update({
        "path": str(path),
        "hash": file_hash or fingerprint_file(path),
        "sample_rate": int(sr),
        "analysis_seconds": round(time.perf_counter() - t0, 3),
    })
    return data
