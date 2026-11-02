"""File names of exported clips: the ``{anime}`` and ``{episode}`` fields.

The naming template (Settings › Export, default
``{anime}_{episode}_scene_{scene_id}``) used to receive the whole file name
for both fields, so every clip carried the episode name twice
(``…ABCD1234_Group_…ABCD1234_scene_468.mp4``). ``split_title`` splits
the usual release names into the series title and the episode number; a
name it can't split goes to ``{anime}`` only.
"""
from __future__ import annotations

import re

# [Group], (1080p), [ABCD1234], {tags}: release group, quality, CRC
_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")
_EPISODE_PATTERNS = (
    # Black Clover S02E04, Title s1e12v2
    re.compile(r"^(?P<anime>.*?)[\s._-]*\b(?P<ep>S\d{1,2}\s*E\d{1,4})(?:v\d)?\b", re.IGNORECASE),
    # Title Episode 12, Title Ep. 12, Title EP12
    re.compile(r"^(?P<anime>.*?)[\s._-]*\b(?:episode|ep)\.?\s*(?P<ep>\d{1,4})(?:v\d)?\b", re.IGNORECASE),
    # Title - 01, Title - Beerus - 01v2 (the last " - <number>")
    re.compile(r"^(?P<anime>.*\S)\s+-\s+(?P<ep>\d{1,4})(?:v\d)?(?:\s|$)"),
    # Title 第12話
    re.compile(r"^(?P<anime>.*?)\s*(?P<ep>第\s*\d{1,4}\s*[話话])"),
)


def split_title(display_name: str) -> tuple[str, str]:
    """``(anime, episode)`` of a video's display name; ``episode`` is "" when
    the name has no recognisable episode number."""
    name = _BRACKETS.sub(" ", display_name or "")
    name = re.sub(r"\s+", " ", name).strip(" -_.")
    for pat in _EPISODE_PATTERNS:
        m = pat.match(name)
        if m:
            anime = m.group("anime").strip(" -_.")
            ep = re.sub(r"\s+", "", m.group("ep"))
            if anime:
                return anime, ep
    return (name or (display_name or "").strip()), ""


def clip_stem(template: str, display_name: str, scene_id: int, tags: str = "") -> str:
    """The template filled in; an empty field takes its separator with it
    (``{anime}_{episode}_scene_…`` without an episode number gives
    ``<anime>_scene_…``)."""
    from ..api.export import _sanitize
    anime, episode = split_title(display_name)
    for field, value in (("episode", episode), ("tags", tags)):
        if not value:
            template = re.sub(r"[_\-. ]?\{" + field + r"\}|\{" + field + r"\}[_\-. ]?", "", template, count=1)
    name = template.format(anime=_sanitize(anime), episode=_sanitize(episode) if episode else "",
                           scene_id=scene_id, tags=tags)
    return name.strip("_-. ") or f"scene_{scene_id}"
