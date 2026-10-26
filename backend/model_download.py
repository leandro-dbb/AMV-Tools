"""Resilient model-weight downloads.

Wraps ``huggingface_hub`` so a flaky connection never leaves the app in a
half-downloaded state without an actionable message:

- retries with exponential backoff;
- verifies file size when the Hub exposes it;
- verifies the SHA-256 when the repo stores the file through LFS;
- keeps Hugging Face's range-request resume (hf_transfer is disabled so the
  10 MB/s bandwidth cap enforced by the project contract still applies).

All helpers raise :class:`ModelDownloadError`, which carries a short ``hint``
for the UI ("check your connection, then press Retry").
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# The shared contract asks for downloads capped at ~10 MB/s. hf_transfer
# parallelises chunks and ignores that cap, so keep it off.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# A stalled connection should fail fast enough to retry instead of hanging.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

try:  # pragma: no cover - exercised against the real Hub in production
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except Exception:  # pragma: no cover
    HfApi = None  # type: ignore[assignment]
    hf_hub_download = None  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]


class ModelDownloadError(RuntimeError):
    def __init__(
        self,
        repo_id: str,
        filename: Optional[str],
        attempts: int,
        cause: BaseException | str,
        hint: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.attempts = attempts
        self.cause = cause
        self.hint = hint or (
            "Check your internet connection and free disk space, then press "
            "Retry. If it keeps failing, a firewall/proxy may be blocking "
            "huggingface.co."
        )
        target = f"{repo_id}/{filename}" if filename else repo_id
        super().__init__(
            f"Could not download {target} after {attempts} attempt(s): {cause}. {self.hint}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "attempts": self.attempts,
            "error": str(self.cause),
            "hint": self.hint,
        }


def _fetch_file_metadata(repo_id: str, token: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """Map filename -> {size, sha256} using the Hub's model metadata.

    Returns an empty dict when metadata is unavailable (offline cache, older
    Hub client, gated repo without token). Downloads still work in that case;
    we just cannot verify the extra properties.
    """
    if HfApi is None:
        return {}
    try:
        info = HfApi().model_info(repo_id, files_metadata=True, token=token)
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    siblings = getattr(info, "siblings", None) or []
    for sibling in siblings:
        name = getattr(sibling, "rfilename", None)
        if not name:
            continue
        entry: dict[str, Any] = {"size": getattr(sibling, "size", None)}
        lfs = getattr(sibling, "lfs", None)
        if lfs is not None:
            entry["sha256"] = getattr(lfs, "sha256", None) or (
                lfs.get("sha256") if isinstance(lfs, dict) else None
            )
        out[name] = entry
    return out


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, metadata: Optional[dict[str, Any]]) -> None:
    if not path.exists():
        raise RuntimeError("downloaded file is missing")
    if not metadata:
        return
    expected_size = metadata.get("size")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise RuntimeError(
            f"size mismatch (expected {expected_size} bytes, got {path.stat().st_size})"
        )
    expected_sha = metadata.get("sha256")
    if expected_sha and _sha256(path) != expected_sha.lower():
        raise RuntimeError("SHA-256 mismatch")


def _backoff(attempt: int, base_delay: float) -> float:
    return min(30.0, base_delay * (2 ** max(0, attempt - 1)))


def download_file(
    repo_id: str,
    filename: str,
    *,
    token: Optional[str] = None,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    metadata: Optional[dict[str, Any]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    if hf_hub_download is None:
        raise ModelDownloadError(repo_id, filename, 0, "huggingface_hub is not installed")

    last_error: BaseException | str = "unknown error"
    force = False
    for attempt in range(1, max_attempts + 1):
        if progress_cb:
            progress_cb(f"Downloading {filename} (attempt {attempt}/{max_attempts})")
        path: Optional[Path] = None
        try:
            # After a failed verification, force a fresh download: the cached
            # copy is the corrupted one (a network error keeps the resume).
            kw = {"force_download": True} if force else {}
            path = Path(hf_hub_download(repo_id, filename, token=token, **kw))
            _verify(path, metadata)
            return str(path)
        except Exception as exc:
            last_error = exc
            # A corrupted/incomplete cached file is the usual cause of a
            # repeated verification failure. Drop it so the next attempt
            # starts from a fresh range request. In the HF cache ``path`` is
            # a snapshot symlink: delete the blob behind it too, or the next
            # attempt relinks the same corrupted bytes.
            if path is not None:
                force = True
                for p in {path.resolve(), path}:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
            if attempt < max_attempts:
                time.sleep(_backoff(attempt, base_delay))
    raise ModelDownloadError(repo_id, filename, max_attempts, last_error)


# ── license allow-list ──────────────────────────────────────────────────────
# AMV Tools is a non-commercial tool: permissive licenses and the Creative
# Commons non-commercial ones are accepted. Weights are never bundled, they
# are fetched on demand from the official source, and Settings shows a
# "non-commercial use" mark next to NC models. OpenRAIL, GPL/AGPL, "other"
# (custom licenses) and unknown licenses are refused here; a custom NC
# license (S-Lab, SAM License) is only accepted for a model the catalogue
# lists explicitly (``NON_COMMERCIAL_LICENSES``).
PERMISSIVE_LICENSES = frozenset({
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0",
})
NON_COMMERCIAL_LICENSES = frozenset({
    "cc-by-nc-4.0", "cc-by-nc-sa-4.0",
})
ALLOWED_LICENSES = PERMISSIVE_LICENSES | NON_COMMERCIAL_LICENSES

_LICENSE_HINT = (
    "AMV Tools only downloads models published under a permissive license "
    "(Apache-2.0, MIT, BSD-2/3-Clause, CC-BY-4.0, CC0-1.0) or a Creative Commons "
    "non-commercial one (CC-BY-NC-4.0, CC-BY-NC-SA-4.0). Pick another model "
    "or tier in Settings > Models."
)


def normalize_license(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) == 1 else None
    if not value or not isinstance(value, str):
        return None
    return value.strip().lower() or None


def license_allowed(value: Any) -> bool:
    return normalize_license(value) in ALLOWED_LICENSES


def is_non_commercial(value: Any) -> bool:
    """True for licenses that forbid commercial use (shown in Settings)."""
    return normalize_license(value) in NON_COMMERCIAL_LICENSES


def is_repo_cached(repo_id: str) -> bool:
    """True when the HF cache already holds a snapshot of ``repo_id`` (no
    network call)."""
    try:
        from huggingface_hub import constants
        root = Path(constants.HF_HUB_CACHE)
    except Exception:
        return False
    snaps = root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    try:
        return snaps.is_dir() and any(snaps.iterdir())
    except OSError:
        return False


def hub_license(repo_id: str, token: Optional[str] = None) -> Optional[str]:
    """License declared on the model card, None when offline/unavailable."""
    if HfApi is None:
        return None
    try:
        info = HfApi().model_info(repo_id, token=token)
    except Exception:
        return None
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    lic = None
    if card is not None:
        lic = getattr(card, "license", None)
        if lic is None and isinstance(card, dict):
            lic = card.get("license")
    return normalize_license(lic)


def catalogue_license(repo_id: str) -> Optional[str]:
    try:
        from .models.tiers import MODEL_INFO
    except Exception:
        return None
    return normalize_license((MODEL_INFO.get(repo_id) or {}).get("license"))


def check_license(repo_id: str, token: Optional[str] = None) -> str:
    """The repo's license if it is on the allow-list, else
    :class:`ModelDownloadError`. The model card wins; offline, the license
    recorded in ``tiers.MODEL_INFO``; unknown is refused."""
    lic = hub_license(repo_id, token) or catalogue_license(repo_id)
    if lic is None:
        raise ModelDownloadError(repo_id, None, 0, "its license could not be determined",
                                 hint=_LICENSE_HINT)
    if lic not in ALLOWED_LICENSES:
        raise ModelDownloadError(repo_id, None, 0, f"license {lic!r} is not on the allow-list",
                                 hint=_LICENSE_HINT)
    return lic


def ensure_license_allowed(repo_id: str, token: Optional[str] = None) -> None:
    """Before the first download of ``repo_id``: :func:`check_license`.
    Already cached models are left alone (no network call)."""
    if is_repo_cached(repo_id):
        return
    check_license(repo_id, token)


def check_finish_license(spec) -> None:
    """Finishing registry models carry their license in ``ModelSpec``."""
    if not license_allowed(spec.license):
        raise ModelDownloadError(spec.id, None, 0, f"license {spec.license!r} is not on the allow-list",
                                 hint=_LICENSE_HINT)


def download_snapshot(
    repo_id: str,
    *,
    token: Optional[str] = None,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    progress_cb: Optional[Callable[[str], None]] = None,
    license_check: bool = True,
    revision: Optional[str] = None,
) -> str:
    if snapshot_download is None:
        raise ModelDownloadError(repo_id, None, 0, "huggingface_hub is not installed")
    if license_check:
        ensure_license_allowed(repo_id, token)

    last_error: BaseException | str = "unknown error"
    for attempt in range(1, max_attempts + 1):
        if progress_cb:
            progress_cb(f"Downloading {repo_id} (attempt {attempt}/{max_attempts})")
        try:
            kw = {"revision": revision} if revision else {}
            return str(snapshot_download(repo_id, token=token, max_workers=1, **kw))
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(_backoff(attempt, base_delay))
    raise ModelDownloadError(repo_id, None, max_attempts, last_error)


def download_model_files(
    repo_id: str,
    filenames: Iterable[str],
    *,
    token: Optional[str] = None,
    max_attempts: int = 5,
    progress_cb: Optional[Callable[[str], None]] = None,
    license_check: bool = True,
) -> list[str]:
    if license_check:
        ensure_license_allowed(repo_id, token)
    metadata = _fetch_file_metadata(repo_id, token)
    paths = []
    for filename in filenames:
        paths.append(
            download_file(
                repo_id,
                filename,
                token=token,
                max_attempts=max_attempts,
                metadata=metadata.get(filename),
                progress_cb=progress_cb,
            )
        )
    return paths


def download_for_key(
    state,
    key: str,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Download the currently configured weights for a diagnostics model key."""
    from .diagnostics import find_model_spec

    spec = find_model_spec(state, key)
    if spec is None:
        return {"ok": False, "error": f"unknown model: {key}", "hint": "Reload the diagnostics panel."}
    repo_id = spec["repo_id"]
    token = (state.settings.get("models", {}).get("hf_token") or None) or None
    # The mask engines keep their own model choice (Settings > Models > Mask).
    license_check = key not in ("birefnet", "sam2")
    revision = None
    if key == "birefnet":
        # same pinned commit the model is loaded at (remote code)
        from .models.birefnet import revision_for_repo
        revision = revision_for_repo(repo_id)
    try:
        if spec.get("files"):
            paths = download_model_files(
                repo_id, spec["files"], token=token, progress_cb=progress_cb,
                license_check=license_check,
            )
        else:
            paths = [download_snapshot(repo_id, token=token, progress_cb=progress_cb,
                                       license_check=license_check, revision=revision)]
    except ModelDownloadError as exc:
        return exc.as_dict()
    return {"ok": True, "repo_id": repo_id, "paths": paths, "hint": ""}


__all__ = [
    "ModelDownloadError",
    "download_file",
    "download_snapshot",
    "download_model_files",
    "download_for_key",
    "ALLOWED_LICENSES",
    "check_license",
    "ensure_license_allowed",
]
