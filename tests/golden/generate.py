"""Regenerate the timeline golden files.

Run from the repo root::

    uv run python tests/golden/generate.py

The golden fixtures intentionally use Windows-style paths so the encoded
``file://`` URLs are identical on every host OS.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.timeline import edl, fcp7, fcpxml, otio  # noqa: E402
from backend.timeline.model import TimelineOptions  # noqa: E402
from tests.test_timeline import _collected_golden_timeline, _golden_timeline  # noqa: E402

OUT = Path(__file__).resolve().parent
WRITERS = {
    "timeline.fcp7.xml": fcp7.dumps,
    "timeline.fcpxml": fcpxml.dumps,
    "timeline.edl": edl.dumps,
    "timeline.otio": otio.dumps,
}


def main() -> None:
    options = TimelineOptions(otio_use_library=False)
    for prefix, timeline in (("timeline", _golden_timeline()),
                             ("timeline.collected", _collected_golden_timeline())):
        for name, writer in WRITERS.items():
            path = OUT / name.replace("timeline", prefix, 1)
            path.write_text(writer(timeline, options), encoding="utf-8")
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
