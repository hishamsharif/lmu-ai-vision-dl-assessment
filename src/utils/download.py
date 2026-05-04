"""
download.py  —  Dataset download utilities
------------------------------------------
Centralises Google Drive file IDs and download/extract logic so the
notebook stays clean and file IDs are versioned alongside the code.

Usage (from notebook):
    from src.utils.download import download_datasets
    download_datasets(dataset_dir='/content/drive/MyDrive/CS7002NU_PPE/datasets')
"""

import os
import shutil
import tempfile
import zipfile
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry — update file IDs here if the Drive files are replaced
# ---------------------------------------------------------------------------

DATASETS = {
    "section_a": {
        "file_id":   "1qhq55YpJr6kLiMiURJtsUEQQkvHtWyKv",
        "zip_name":  "section_a.zip",
        "dest_dir":  "section_a",
        # Guard: skip download if at least one .jpeg already present
        "is_ready":  lambda dest: any(
            f.lower().endswith(('.jpeg', '.jpg', '.png'))
            for f in os.listdir(dest)
        ) if os.path.isdir(dest) else False,
    },
    "section_b": {
        "file_id":   "1EAN7Ck2B1SdwUIPisBIe-QPPZ0UKbNoV",
        "zip_name":  "section_b.zip",
        "dest_dir":  "section_b",
        # Guard: skip download if train/ exists at either expected depth:
        #   section_b/raw/train/  (expected)
        #   section_b/train/      (fallback if zip has no raw/ level)
        "is_ready":  lambda dest: (
            os.path.isdir(os.path.join(dest, "raw", "train")) or
            os.path.isdir(os.path.join(dest, "train"))
        ),
    },
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ensure_gdown():
    """Import gdown, installing it quietly if not already available."""
    try:
        import gdown
        return gdown
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
        return gdown


def _download_and_extract(file_id: str, zip_name: str, dest_dir: str) -> None:
    """
    Download a public Google Drive zip and place its contents at dest_dir.

    dest_dir is the FINAL destination folder (e.g. dataset_dir/section_a/).
    The function handles both zip structures transparently:
      - zip with a single root folder  → that folder's contents land in dest_dir
      - flat zip (no root folder)      → all files land directly in dest_dir
    Any pre-existing content at dest_dir is removed first to avoid stale
    double-nested trees from previous extractions.
    """
    gdown = _ensure_gdown()

    tmp_zip = f"/tmp/{zip_name}"
    logger.info("Downloading %s (id=%s) ...", zip_name, file_id)
    gdown.download(id=file_id, output=tmp_zip, quiet=False)

    logger.info("Extracting %s -> %s", zip_name, dest_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(tmp_zip, "r") as z:
            z.extractall(tmp_dir)

        # Detect whether the zip had a single root folder.
        entries = os.listdir(tmp_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
            src = os.path.join(tmp_dir, entries[0])   # unwrap the root folder
        else:
            src = tmp_dir                              # flat zip — use tmp_dir directly

        # Remove stale destination (handles leftover double-nested trees).
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)

        shutil.copytree(src, dest_dir)

    os.remove(tmp_zip)
    logger.info("%s ready.", zip_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_datasets(dataset_dir: str, subsets: list = None) -> dict:
    """
    Download and extract all registered datasets into dataset_dir.

    Args:
        dataset_dir: Root folder where each dataset subdirectory will be created.
                     Typically Drive/CS7002NU_PPE/datasets/
        subsets:     List of dataset keys to download, e.g. ['section_a'].
                     Defaults to all registered datasets.

    Returns:
        Dict mapping dataset key -> absolute path of its extracted directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    keys   = subsets or list(DATASETS.keys())
    paths  = {}

    for key in keys:
        if key not in DATASETS:
            raise ValueError(f"Unknown dataset '{key}'. Available: {list(DATASETS.keys())}")

        cfg      = DATASETS[key]
        dest_dir = os.path.join(dataset_dir, cfg["dest_dir"])

        if cfg["is_ready"](dest_dir):
            print(f"[{key}] already on Drive — skipping download")
        else:
            _download_and_extract(cfg["file_id"], cfg["zip_name"], dest_dir)
            print(f"[{key}] extracted -> {dest_dir}")

        paths[key] = dest_dir

    return paths
