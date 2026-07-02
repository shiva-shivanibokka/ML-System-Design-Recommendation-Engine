"""
Startup artifact downloader.

Pulls the trained model + processed-data bundle from a Hugging Face Hub model
repo into the local paths the gateway expects (models/, data/). Enables a
stateless container (HF Space) to serve real models without baking them into
the image. Degrades gracefully: if the repo is unset or the download fails,
returns False and the app falls back to cold-start/popularity behavior.
"""
from __future__ import annotations

from pathlib import Path

import structlog

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - import guard for minimal envs
    snapshot_download = None  # type: ignore

log = structlog.get_logger()

# Repo root = two levels up from this file (serving/artifacts.py -> repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def download_artifacts(repo_id: str | None, token: str | None = None) -> bool:
    """Download the artifact bundle from HF Hub into the repo root.

    Returns True if a download was attempted successfully, False if skipped
    (no repo_id) or on any failure.
    """
    if not repo_id:
        log.info("artifacts.skip_no_repo")
        return False
    if snapshot_download is None:
        log.warning("artifacts.hf_hub_unavailable")
        return False
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            local_dir=str(_REPO_ROOT),
            allow_patterns=["models/**", "data/indexes/**", "data/processed/**"],
        )
        log.info("artifacts.downloaded", repo_id=repo_id)
        return True
    except Exception as e:
        log.warning("artifacts.download_failed", repo_id=repo_id, error=str(e))
        return False
