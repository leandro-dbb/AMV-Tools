"""AMV Tools backend — FastAPI sidecar for the Electron app."""
__version__ = "0.5.0b1"

try:  # see backend.av_quiet: an HEVC decode could deadlock the whole sidecar
    from .av_quiet import install as _install_av_quiet
    _install_av_quiet()
except ImportError:  # PyAV missing: nothing decodes anyway
    pass
