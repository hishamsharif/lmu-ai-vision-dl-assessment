"""
download.py  —  Dataset & asset download utilities
---------------------------------------------------
Centralises all Google Drive file / folder IDs and download logic so the
notebook stays clean and IDs are versioned alongside the code.

All public Drive links are accessed without authentication via gdown.

Usage (from notebook):
    from src.utils.download import download_datasets, download_section_b_assets

    # Section A images + Section B YOLO dataset (zip files → datasets/)
    paths = download_datasets(dataset_dir)

    # Section B pre-trained model + pre-computed figures (shared folder → section_b/)
    b = download_section_b_assets(dataset_dir)
    ASSET_DIR = b['asset_dir']   # parent of figures/
    MODEL_DIR = b['model_dir']   # directory holding *.keras files
    DATA_DIR  = b['data_dir']    # COCO dataset root (None → Roboflow fallback)
"""

import os
import shutil
import tempfile
import zipfile
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zip-based dataset registry (Section A images, Section B YOLO dataset)
# ---------------------------------------------------------------------------

DATASETS = {
    "section_a": {
        "file_id":  "1qhq55YpJr6kLiMiURJtsUEQQkvHtWyKv",
        "zip_name": "section_a.zip",
        "dest_dir": "section_a",
        "is_ready": lambda dest: any(
            f.lower().endswith(('.jpeg', '.jpg', '.png'))
            for f in os.listdir(dest)
        ) if os.path.isdir(dest) else False,
    },
    "section_b": {
        "file_id":  "1EAN7Ck2B1SdwUIPisBIe-QPPZ0UKbNoV",
        "zip_name": "section_b.zip",
        "dest_dir": "section_b",
        "is_ready": lambda dest: (
            os.path.isdir(os.path.join(dest, "raw", "train")) or
            os.path.isdir(os.path.join(dest, "train"))
        ),
    },
}

# ---------------------------------------------------------------------------
# Shared folder — Section B pre-trained model + pre-computed figures
# Shared by Udam / Tharinda (publicly accessible, no authentication required).
# Section A images (imgA/B/C.jpg) present in the folder are skipped — already
# in datasets/section_a/.
# ---------------------------------------------------------------------------

_SECTION_B_ASSETS_FOLDER_ID = "1gPgz8HdcNfSIT62ooKn2wZJUSiwV1tkr"

# Files in the shared folder that duplicate Section A content — skip them.
_SKIP_FILENAMES = {"imgA.jpg", "imgB.jpg", "imgC.jpg"}

# Best model checkpoint name (used as the ready-guard).
BEST_MODEL_NAME = "R6_S_oneCycle_weighted_best.keras"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_gdown():
    """Import gdown, installing it quietly if not already available."""
    try:
        import gdown
        return gdown
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"],
                       check=True)
        import gdown
        return gdown


def _download_and_extract(file_id: str, zip_name: str, dest_dir: str) -> None:
    """
    Download a public Google Drive zip and place its contents at dest_dir.

    Handles both zip structures:
      - single root folder inside zip  → contents land directly in dest_dir
      - flat zip                       → files land directly in dest_dir
    Pre-existing dest_dir is removed to avoid stale nested trees.
    """
    gdown = _ensure_gdown()

    tmp_zip = f"/tmp/{zip_name}"
    logger.info("Downloading %s (id=%s) ...", zip_name, file_id)
    gdown.download(id=file_id, output=tmp_zip, quiet=False)

    logger.info("Extracting %s -> %s", zip_name, dest_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(tmp_zip, "r") as z:
            z.extractall(tmp_dir)

        entries = os.listdir(tmp_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
            src = os.path.join(tmp_dir, entries[0])
        else:
            src = tmp_dir

        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(src, dest_dir)

    os.remove(tmp_zip)
    logger.info("%s ready.", zip_name)


def _download_folder_filtered(folder_id: str, dest_dir: str,
                               skip_files: set) -> None:
    """
    Download a public Google Drive folder into dest_dir, with two behaviours:
      - Files whose basenames are in skip_files are not copied.
      - .zip files are extracted in-place (e.g. notebook_assets.zip →
        dest_dir/notebook_assets/) rather than copied as archives.

    Uses a temporary directory so dest_dir is only written after the full
    download completes.  Existing content in dest_dir is preserved (merge).
    """
    gdown = _ensure_gdown()

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info("Downloading Drive folder %s ...", folder_id)
        # gdown creates a subfolder named after the Drive folder inside tmp_dir.
        gdown.download_folder(id=folder_id, output=tmp_dir, quiet=False)

        # Unwrap single root subfolder if present.
        entries = os.listdir(tmp_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
            src_root = os.path.join(tmp_dir, entries[0])
        else:
            src_root = tmp_dir

        os.makedirs(dest_dir, exist_ok=True)
        skipped, extracted = [], []

        for item in os.listdir(src_root):
            if item in skip_files:
                skipped.append(item)
                continue

            src = os.path.join(src_root, item)

            if item.lower().endswith(".zip"):
                # Extract notebook_assets.zip → dest_dir/notebook_assets/
                stem       = os.path.splitext(item)[0]
                extract_to = os.path.join(dest_dir, stem)
                if os.path.exists(extract_to):
                    shutil.rmtree(extract_to)
                with zipfile.ZipFile(src, "r") as z:
                    z.extractall(extract_to)
                extracted.append(f"{item} -> {stem}/")
            elif os.path.isdir(src):
                dst = os.path.join(dest_dir, item)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, os.path.join(dest_dir, item))

        if skipped:
            print(f"[section_b_assets] skipped (already in section_a): {', '.join(skipped)}")
        if extracted:
            print(f"[section_b_assets] extracted: {', '.join(extracted)}")
        logger.info("Folder contents merged into %s", dest_dir)


def _print_tree(base_dir: str, max_depth: int = 3) -> None:
    """Print the directory tree under base_dir up to max_depth levels."""
    base_depth = base_dir.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(base_dir):
        depth = root.count(os.sep) - base_depth
        if depth > max_depth:
            dirs.clear()
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or root}/")
        sub = "  " * (depth + 1)
        for f in sorted(files)[:10]:
            print(f"{sub}{f}")
        if len(files) > 10:
            print(f"{sub}... ({len(files) - 10} more files)")


def _walk_find_asset_dir(base_dir: str) -> str | None:
    """
    Return the ASSET_DIR — the directory that directly contains figures/*.png.

    Checks two layouts:
      1. <dir>/figures/*.png  (standard: notebook_assets/figures/)
      2. <dir>/*.png          (flat: PNGs stored directly in a directory)
    """
    for root, dirs, _ in os.walk(base_dir):
        if "figures" in dirs:
            fig_path = os.path.join(root, "figures")
            if any(f.endswith(".png") for f in os.listdir(fig_path)):
                return root
    # Fallback: flat layout
    for root, _, files in os.walk(base_dir):
        if sum(1 for f in files if f.endswith(".png")) >= 3:
            return root
    return None


def _walk_find_model_dir(base_dir: str) -> str | None:
    """Return the first directory that contains at least one .keras file."""
    for root, _, files in os.walk(base_dir):
        if any(f.endswith(".keras") for f in files):
            return root
    return None


def _walk_find_data_dir(base_dir: str) -> str | None:
    """
    Return the COCO dataset root — directory containing train/ with
    _annotations.coco.json.
    """
    for root, dirs, _ in os.walk(base_dir):
        if "train" in dirs:
            ann = os.path.join(root, "train", "_annotations.coco.json")
            if os.path.isfile(ann):
                return root
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_datasets(dataset_dir: str, subsets: list = None) -> dict:
    """
    Download and extract Section A / B zip datasets into dataset_dir.

    Args:
        dataset_dir: Root folder for datasets (e.g. Drive/CS7002NU_PPE/datasets/).
        subsets:     Keys to download, e.g. ['section_a'].  Defaults to all.

    Returns:
        Dict mapping dataset key -> absolute path of its extracted directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    keys  = subsets or list(DATASETS.keys())
    paths = {}

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


def download_section_b_assets(dataset_dir: str) -> dict:
    """
    Download Section B pre-trained model and pre-computed figures from the
    team's shared public Google Drive folder into dataset_dir/section_b/.

    Content is merged alongside the existing YOLO dataset (raw/).
    Section A images (imgA/B/C.jpg) present in the shared folder are skipped.
    The COCO dataset is NOT included in the shared folder; DATA_DIR will be
    None and the Roboflow fallback in the notebook will handle it.

    Args:
        dataset_dir: datasets/ root (e.g. Drive/CS7002NU_PPE/datasets/).

    Returns:
        dict with keys:
            asset_dir   — parent of figures/  (set as ASSET_DIR in notebook)
            figures_dir — directory holding *.png files
            model_dir   — directory holding *.keras files  (set as MODEL_DIR)
            data_dir    — COCO dataset root, or None
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    section_b_dir = os.path.join(dataset_dir, "section_b")

    # Guard: skip download if the best model is already present in section_b/
    model_dir = _walk_find_model_dir(section_b_dir)
    if model_dir is not None:
        print(f"[section_b_assets] already in {section_b_dir} — skipping download")
    else:
        print(f"[section_b_assets] downloading shared folder into {section_b_dir} ...")
        _download_folder_filtered(
            _SECTION_B_ASSETS_FOLDER_ID,
            section_b_dir,
            skip_files=_SKIP_FILENAMES,
        )
        model_dir = _walk_find_model_dir(section_b_dir)

    # ── Diagnostic tree ───────────────────────────────────────────────────────
    print(f"\n[section_b_assets] section_b/ tree:")
    _print_tree(section_b_dir, max_depth=3)

    # ── Detect paths ──────────────────────────────────────────────────────────
    asset_dir = _walk_find_asset_dir(section_b_dir)
    data_dir  = _walk_find_data_dir(section_b_dir)

    figures_subdir = None
    if asset_dir is not None:
        candidate = os.path.join(asset_dir, "figures")
        figures_subdir = candidate if os.path.isdir(candidate) else asset_dir

    print(f"\n[section_b_assets] asset_dir   -> {asset_dir    or '(not found)'}")
    print(f"[section_b_assets] figures_dir -> {figures_subdir or '(not found)'}")
    print(f"[section_b_assets] model_dir   -> {model_dir     or '(not found)'}")
    print(f"[section_b_assets] data_dir    -> {data_dir      or '(not found — Roboflow fallback will run)'}")

    return {
        "asset_dir":   asset_dir    or section_b_dir,
        "figures_dir": figures_subdir or section_b_dir,
        "model_dir":   model_dir    or section_b_dir,
        "data_dir":    data_dir,
    }
