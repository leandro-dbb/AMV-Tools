"""Music sync: audio analysis, beat-synced shot proposals, exports, preview.

The package is deliberately independent from FastAPI and SQLite (except
:mod:`.library`, which reads the shots) so the analysis and the planner can
be unit-tested on synthetic signals.  The API layer lives in
:mod:`backend.api.music`.

Licences of the dependencies used here (all compatible with the MIT repo
licence, all redistributable):

* librosa 1.0 — ISC (pulls numba/llvmlite BSD, scikit-learn BSD-3,
  soundfile BSD-3, soxr LGPL-2.1+ as a separate dynamic module)
* numpy / scipy — BSD-3-Clause
* ffmpeg (bundled imageio-ffmpeg build, already used for video) — decoding
  and the preview render.

No model weights are downloaded for music analysis: everything is signal
processing on the CPU, which keeps the feature available on every machine
(Windows/NVIDIA users included) without a GPU.
"""
from .analysis import ANALYSIS_VERSION, analyze_audio, analyze_signal
from .audio import AudioDecodeError, decode_audio, fingerprint_file

__all__ = [
    "ANALYSIS_VERSION",
    "AudioDecodeError",
    "analyze_audio",
    "analyze_signal",
    "decode_audio",
    "fingerprint_file",
]
