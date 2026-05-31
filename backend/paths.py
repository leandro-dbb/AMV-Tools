"""Resolve user-data paths in a way that works both in dev and bundled."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Where settings, default DBs, proxies, exports live."""
    forced = os.environ.get("AMV_DATA_DIR")
    if forced:
        p = Path(forced)
    elif sys.platform == "win32":
        p = Path(os.environ.get("APPDATA", str(Path.home()))) / "AMVTools"
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / "AMVTools"
    else:
        p = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "amv-tools"
    p.mkdir(parents=True, exist_ok=True)
    (p / "proxies").mkdir(exist_ok=True)
    (p / "exports").mkdir(exist_ok=True)
    (p / "thumbs").mkdir(exist_ok=True)
    (p / "logs").mkdir(exist_ok=True)
    return p


def default_db_path() -> Path:
    return user_data_dir() / "amv_tools.db"


def settings_path() -> Path:
    return user_data_dir() / "settings.json"
