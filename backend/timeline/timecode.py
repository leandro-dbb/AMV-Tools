"""Exact frame/timecode maths for the timeline exporters.

Everything internal is a :class:`fractions.Fraction` so 23.976/29.97 never
drift: 23.976 is 24000/1001, not ``23.976``.  Timecode strings still use the
nominal integer rate (24, 30, 60) as required by CMX3600 and the NLEs.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Union

Number = Union[int, float, str, Fraction]

# Values NLEs commonly report as floating point.  Map them back to the exact
# NTSC rational so frame conversion stays lossless.
_NTSC_ALIASES = {
    23.976: Fraction(24000, 1001),
    23.98: Fraction(24000, 1001),
    29.97: Fraction(30000, 1001),
    29.98: Fraction(30000, 1001),
    59.94: Fraction(60000, 1001),
    59.97: Fraction(60000, 1001),
    119.88: Fraction(120000, 1001),
}


def parse_rate(value: Number | None) -> Fraction | None:
    """Parse ``"24000/1001"``, ``29.97``, ``Fraction(...)`` or ``None``."""
    if value is None or value == "":
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        for alias, exact in _NTSC_ALIASES.items():
            if abs(value - alias) < 0.001:
                return exact
        return Fraction(value).limit_denominator(1_000_000)
    text = str(value).strip()
    if not text or text.lower() == "first":
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        return Fraction(int(num), int(den))
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"invalid frame rate: {value!r}") from exc
    return parse_rate(number)


def nominal_rate(rate: Fraction) -> int:
    """Integer timecode rate: 23.976 -> 24, 29.97 -> 30, 59.94 -> 60."""
    return max(1, int(round(float(rate))))


def is_ntsc(rate: Fraction) -> bool:
    return rate.denominator == 1001 or abs(float(rate) - nominal_rate(rate)) > 0.001


def is_drop_frame_rate(rate: Fraction) -> bool:
    """Only 29.97 and 59.94 (and their 1000/1001 multiples) drop frames."""
    return is_ntsc(rate) and nominal_rate(rate) in (30, 60, 120)


def drop_frames_per_minute(rate: Fraction) -> int:
    nominal = nominal_rate(rate)
    # 2 at 29.97, 4 at 59.94 — round(nominal * 0.066666…)
    return nominal // 15


def _df_add_dropped(frame: int, rate: Fraction) -> int:
    """Convert a real frame count into the drop-frame number line."""
    nominal = nominal_rate(rate)
    drop = drop_frames_per_minute(rate)
    frames_per_10_min = nominal * 60 * 10 - drop * 9
    frames_per_min = nominal * 60 - drop
    ten_min, rem = divmod(frame, frames_per_10_min)
    if rem >= drop:
        extra = ((rem - drop) // frames_per_min) * drop
    else:
        extra = 0
    return frame + drop * 9 * ten_min + extra


def _df_remove_dropped(frame: int, rate: Fraction) -> int:
    """Inverse of :func:`_df_add_dropped`: ``frame`` is a drop-frame label
    count (``(hh*60+mm)*60+ss)*nominal+ff``); every minute not divisible by
    10 skipped ``drop`` labels."""
    nominal = nominal_rate(rate)
    drop = drop_frames_per_minute(rate)
    total_min = frame // (nominal * 60)
    return frame - drop * (total_min - total_min // 10)


def frames_to_timecode(frame: int, rate: Fraction, drop_frame: bool = False) -> str:
    """Render a frame number as ``HH:MM:SS:FF`` (or ``;FF`` in drop-frame)."""
    frame = max(0, int(frame))
    nominal = nominal_rate(rate)
    drop = bool(drop_frame and is_drop_frame_rate(rate))
    if drop:
        frame = _df_add_dropped(frame, rate)
    sign = ""
    ff = frame % nominal
    total_seconds = frame // nominal
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = (total_seconds // 3600) % 24
    sep = ";" if drop else ":"
    return f"{sign}{hh:02d}:{mm:02d}:{ss:02d}{sep}{ff:02d}"


def timecode_to_frames(tc: str, rate: Fraction, drop_frame: bool = False) -> int:
    """Inverse of :func:`frames_to_timecode`."""
    text = tc.strip().replace(".", ":").replace(";", ":")
    parts = text.split(":")
    if len(parts) != 4:
        raise ValueError(f"invalid timecode: {tc!r}")
    hh, mm, ss, ff = (int(p) for p in parts)
    nominal = nominal_rate(rate)
    if ff >= nominal:
        raise ValueError(f"frame field {ff} out of range for {nominal} fps")
    frame = ((hh * 60 + mm) * 60 + ss) * nominal + ff
    drop = bool(drop_frame and is_drop_frame_rate(rate))
    if drop:
        frame = _df_remove_dropped(frame, rate)
    return frame


def frames_to_seconds(frames: int, rate: Fraction) -> Fraction:
    """Exact duration in seconds of ``frames`` frames at ``rate``."""
    return Fraction(int(frames)) / rate


def seconds_to_frames(seconds: Number, rate: Fraction) -> int:
    return int(round(float(Fraction(seconds) * rate)))


def ms_to_frames(ms: Number, rate: Fraction) -> int:
    return seconds_to_frames(Fraction(ms, 1000), rate)


def frames_to_ms(frames: int, rate: Fraction) -> int:
    return int(round(float(frames_to_seconds(frames, rate)) * 1000))


def convert_frames(frames: int, src_rate: Fraction, dst_rate: Fraction) -> int:
    """Rescale a frame count from one rate to another, rounded to nearest."""
    if src_rate == dst_rate:
        return int(frames)
    return int(round(float(Fraction(int(frames)) * dst_rate / src_rate)))


def fps_decimal(rate: Fraction) -> float:
    return float(rate)


def rate_string(rate: Fraction) -> str:
    """Human/JSON friendly exact rate, e.g. ``"24000/1001"`` or ``"25"``."""
    if rate.denominator == 1:
        return str(rate.numerator)
    return f"{rate.numerator}/{rate.denominator}"


def fps_label(rate: Fraction) -> str:
    value = float(rate)
    for known in (23.976, 29.97, 59.94, 119.88):
        if abs(value - known) < 0.001:
            return f"{known:g}"
    return f"{value:g}"


def fcpxml_time(frames: int, rate: Fraction) -> str:
    """FCPXML rational time for ``frames`` frames at ``rate``."""
    return _fcpxml_fraction(Fraction(int(frames)) / rate)


def fcpxml_seconds(seconds: Fraction) -> str:
    """FCPXML rational time for an exact number of seconds."""
    return _fcpxml_fraction(Fraction(seconds))


def fcpxml_frame_duration(rate: Fraction) -> str:
    return _fcpxml_fraction(Fraction(1, 1) / rate)


def _fcpxml_fraction(seconds: Fraction) -> str:
    if seconds.denominator == 1:
        return f"{seconds.numerator}s"
    return f"{seconds.numerator}/{seconds.denominator}s"


def otio_rate(rate: Fraction) -> float:
    """OTIO stores rates as a double, exactly what the reference library does."""
    return float(rate)
