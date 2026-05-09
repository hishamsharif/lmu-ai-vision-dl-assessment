"""
download.py  —  Dataset & asset download utilities
---------------------------------------------------
Centralises all Google Drive file / folder IDs and download logic so the
notebook stays clean and IDs are versioned alongside the code.

All public Drive links are accessed without authentication via gdown.

Usage (from notebook):
    from src.utils.download import download_datasets, download_section_b_assets

    # Section A + B YOLO datasets (zip files)
    paths = download_datasets(dataset_dir)

    # Section B pre-trained models + pre-computed figures + COCO dataset
    b = download_section_b_assets(drive_root)
    ASSET_DIR = b['asset_dir']   # figures/
    MODEL_DIR = b['model_dir']   # .keras files
    DATA_DIR  = b['data_dir']    # COCO train/valid/test
"""

import os
import shutil
import tempfile
import zipfile
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zip-based dataset registry (Section A images, Section B YOLO dataset)
# Update file IDs here if the Drive files are replaced.
# ---------------------------------------------------------------------------

DATASETS = {
    "section_a": {
        "file_id":  "1qhq55YpJr6kLiMiURJtsUEQQkvHtWyKv",
        "zip_name": "section_a.zip",
        "dest_dir": "section_a",
        # Guard: skip if at least one image is already present.
        "is_ready": lambda dest: any(
            f.lower().endswith(('.jpeg', '.jpg', '.png'))
            for f in os.listdir(dest)
        ) if os.path.isdir(dest) else False,
    },
    "section_b": {
        "file_id":  "1EAN7Ck2B1SdwUIPisBIe-QPPZ0UKbNoV",
        "zip_name": "section_b.zip",
        "dest_dir": "section_b",
        # Guard: skip if train/ exists at either expected depth.
        "is_ready": lambda dest: (
            os.path.isdir(os.path.join(dest, "raw", "train")) or
            os.path.isdir(os.path.join(dest, "train"))
        ),
    },
}

# ---------------------------------------------------------------------------
# Folder-based asset registry (Section B pre-trained models + figures + COCO)
# Shared by Udam / Tharinda — publicly accessible without authentication.
# ---------------------------------------------------------------------------

# Top-level Google Drive folder ID for all Section B assets.
_SECTION_B_ASSETS_FOLDER_ID = "1gPgz8HdcNfSIT62ooKn2wZJUSiwV1tkr"

# Name of the best model checkpoint expected inside MODEL_DIR.
BEST_MODEL_NAME = "R6_S_oneCycle_weighted_best.keras"

# Expected figure filenames inside ASSET_DIR/figures/.
EXPECTED_FIGURES = [
    "class_distribution.png",
    "training_curves_all.png",
    "training_curves_top3.png",
    "confusion_matrices.png",
    "roc_curves.png",
    "sample_predictions.png",
]


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

    Handles both zip structures transparently:
      - zip with a single root folder  → that folder's contents land in dest_dir
      - flat zip (no root folder)      → all files land directly in dest_dir
    Pre-existing content at dest_dir is removed first to avoid stale nested trees.
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
            src = os.path.join(tmp_dir, entries[0])   # unwrap single root folder
        else:
            src = tmp_dir                              # flat zip

        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(src, dest_dir)

    os.remove(tmp_zip)
    logger.info("%s ready.", zip_name)


def _download_folder(folder_id: str, dest_dir: str) -> None:
    """
    Download an entire public Google Drive folder into dest_dir using gdown.

    gdown.download_folder recreates the Drive folder structure under dest_dir,
    so dest_dir becomes the parent of whatever top-level directories the Drive
    folder contains (e.g. dest_dir/ppe_assets/, dest_dir/ppe_models/, ...).
    """
    gdown = _ensure_gdown()
    os.makedirs(dest_dir, exist_ok=True)
    logger.info("Downloading Drive folder %s -> %s", folder_id, dest_dir)
    gdown.download_folder(id=folder_id, output=dest_dir, quiet=False)
    logger.info("Folder download complete -> %s", dest_dir)


def _print_tree(base_dir: str, max_depth: int = 3) -> None:
    """Print the directory tree under base_dir up to max_depth levels."""
    base_depth = base_dir.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(base_dir):
        depth = root.count(os.sep) - base_depth
        if depth > max_depth:
            dirs.clear()
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        sub = "  " * (depth + 1)
        for f in sorted(files)[:10]:          # cap at 10 files per dir
            print(f"{sub}{f}")
        if len(files) > 10:
            print(f"{sub}... ({len(files) - 10} more files)")


def _walk_find_asset_dir(base_dir: str) -> str | None:
    """
    Return the directory that acts as ASSET_DIR — the parent of figures/*.png.

    Checks two layouts:
      1. <dir>/figures/*.png   (expected: ppe_assets/figures/)
      2. <dir>/*.png           (flat: PNG files directly in a directory)

    Returns the directory such that os.path.join(asset_dir, 'figures', '*.png')
    resolves to the figures.  For the flat case, a 'figures' symlink is NOT
    created; callers should check both ASSET_DIR/figures/ and ASSET_DIR/ itself.
    """
    # Priority 1: proper figures/ subdirectory
    for root, dirs, _ in os.walk(base_dir):
        if "figures" in dirs:
            fig_path = os.path.join(root, "figures")
            if any(f.endswith(".png") for f in os.listdir(fig_path)):
                return root

    # Priority 2: PNG files stored flat inside any subdirectory
    for root, _, files in os.walk(base_dir):
        pngs = [f for f in files if f.endswith(".png")]
        if len(pngs) >= 3:       # at least 3 figures present
            return root          # caller treats this dir as both ASSET_DIR and figures/

    return None


def _walk_find_model_dir(base_dir: str) -> str | None:
    """Return the first directory that contains at least one .keras file."""
    for root, _, files in os.walk(base_dir):
        if any(f.endswith(".keras") for f in files):
            return root
    return None


def _walk_find_data_dir(base_dir: str) -> str | None:
    """
    Return the COCO dataset root — the directory that has train/, valid/, test/
    subdirectories each containing '_annotations.coco.json'.
    """
    for root, dirs, _ in os.walk(base_dir):
        if "train" in dirs:
            ann = os.path.join(root, "train", "_annotations.coco.json")
            if os.path.isfile(ann):
                return root
    return None


def _section_b_assets_ready(dest_dir: str) -> bool:
    """
    True if dest_dir is non-empty (download already ran).
    We re-run detection each time so path resolution stays accurate even if
    the folder structure changed between sessions.
    """
    return os.path.isdir(dest_dir) and bool(os.listdir(dest_dir))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_datasets(dataset_dir: str, subsets: list = None) -> dict:
    """
    Download and extract Section A / B zip datasets into dataset_dir.

    Args:
        dataset_dir: Root folder for datasets (typically Drive/CS7002NU_PPE/datasets/).
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


def download_section_b_assets(drive_root: str) -> dict:
    """
    Download Section B pre-trained models, pre-computed figures, and COCO dataset
    from the team's shared public Google Drive folder.

    All content is saved under drive_root/section_b_assets/ so it persists
    across Colab sessions.  A guard check skips the download if the assets
    are already present.

    Args:
        drive_root: Top-level Drive directory (e.g. /content/drive/MyDrive/CS7002NU_PPE).

    Returns:
        dict with keys:
            asset_dir  — parent of figures/  (pass as ASSET_DIR in notebook)
            model_dir  — directory holding *.keras files  (pass as MODEL_DIR)
            data_dir   — COCO dataset root with train/valid/test  (pass as DATA_DIR)
                         None if the shared folder does not include the dataset.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Search for an already-present section_b_assets folder in two locations:
    #   1. drive_root/section_b_assets   (preferred — inside the project folder)
    #   2. parent(drive_root)/section_b_assets  (MyDrive root — common when the
    #      folder was added to Drive via the sharing UI rather than downloaded)
    candidate_paths = [
        os.path.join(drive_root, "section_b_assets"),
        os.path.join(os.path.dirname(drive_root), "section_b_assets"),
    ]

    dest = None
    for path in candidate_paths:
        if _section_b_assets_ready(path):
            dest = path
            print(f"[section_b_assets] found existing assets at {dest}")
            break

    if dest is None:
        dest = candidate_paths[0]          # download into the project folder
        print("[section_b_assets] downloading from shared Drive folder ...")
        _download_folder(_SECTION_B_ASSETS_FOLDER_ID, dest)

    # ── Diagnostic: show what was downloaded ─────────────────────────────────
    print("\n[section_b_assets] folder tree:")
    _print_tree(dest, max_depth=3)

    # ── Detect paths ──────────────────────────────────────────────────────────
    asset_dir = _walk_find_asset_dir(dest)
    model_dir = _walk_find_model_dir(dest)
    data_dir  = _walk_find_data_dir(dest)

    # If PNGs live flat (no figures/ subdir), treat that dir as figures/ itself
    # by setting FIGURES_DIR = asset_dir so callers can check both.
    figures_subdir = None
    if asset_dir is not None:
        candidate = os.path.join(asset_dir, "figures")
        if os.path.isdir(candidate):
            figures_subdir = candidate
        else:
            figures_subdir = asset_dir   # flat layout — PNGs are directly here

    print(f"\n[section_b_assets] asset_dir    -> {asset_dir   or '(not found)'}")
    print(f"[section_b_assets] figures_dir  -> {figures_subdir or '(not found)'}")
    print(f"[section_b_assets] model_dir    -> {model_dir   or '(not found)'}")
    print(f"[section_b_assets] data_dir     -> {data_dir    or '(not found — Roboflow fallback will run)'}")

    return {
        "asset_dir":   asset_dir   or dest,
        "figures_dir": figures_subdir or dest,
        "model_dir":   model_dir   or dest,
        "data_dir":    data_dir,            # None → Roboflow fallback in notebook
    }
