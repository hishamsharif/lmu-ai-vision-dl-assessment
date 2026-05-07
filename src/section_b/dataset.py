"""
dataset.py  —  Section B: Classification Dataset from YOLO Annotations
-----------------------------------------------------------------------
Converts a YOLOv8-format dataset (images + .txt label files) into a
TensorFlow classification dataset by cropping each bounding box region.

Pipeline
--------
1. Read data.yaml  → class_names list
2. For each image + label file pair in each split:
   - Load image with OpenCV
   - Parse YOLO box: class_id  cx  cy  w  h  (all normalised to [0,1])
   - Crop the box region; skip degenerate (zero-area) boxes
   - Resize crop to img_size × img_size; normalise to [0, 1]
3. Stratified 80 / 10 / 10 train/val/test split
4. Build augmented tf.data.Dataset pipelines

Public API
----------
load_yolo_crops(dataset_dir, splits=('train','valid'), img_size=224)
    -> X (np.ndarray float32), y (np.ndarray int32), class_names (list[str])

make_tf_datasets(X, y, num_classes, val_split=0.10, test_split=0.10,
                 batch_size=32, augment=True, seed=42)
    -> train_ds, val_ds, test_ds, (X_test, y_test)

visualise_samples(X, y, class_names, n=12, output_path=None) -> None
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_data_yaml(dataset_dir: str) -> list:
    """Return class_names list from data.yaml.  Searches dataset_dir and its parent."""
    try:
        import yaml
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyyaml'], check=True)
        import yaml

    candidates = [
        os.path.join(dataset_dir, 'data.yaml'),
        os.path.join(os.path.dirname(dataset_dir), 'data.yaml'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get('names', [])
            if isinstance(names, dict):          # YOLO sometimes uses {0: 'cat', 1: 'dog'}
                names = [names[k] for k in sorted(names)]
            return names

    raise FileNotFoundError(
        f'data.yaml not found near {dataset_dir}. '
        'Checked: ' + ', '.join(candidates)
    )


def _crop_box(img, cx_n: float, cy_n: float, w_n: float, h_n: float,
              img_size: int):
    """
    Crop a normalised YOLO bounding box from img and resize to img_size×img_size.
    Returns None if the crop is degenerate (zero area or out of bounds).
    """
    h, w = img.shape[:2]
    x1 = max(0, int((cx_n - w_n / 2) * w))
    y1 = max(0, int((cy_n - h_n / 2) * h))
    x2 = min(w, int((cx_n + w_n / 2) * w))
    y2 = min(h, int((cy_n + h_n / 2) * h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_yolo_crops(dataset_dir: str,
                    splits: tuple = ('train', 'valid'),
                    img_size: int = 224) -> tuple:
    """
    Load and crop all bounding-box regions from a YOLO-format dataset.

    Args:
        dataset_dir: Root of the YOLO dataset.  Must contain data.yaml and
                     {split}/images/ + {split}/labels/ subdirectories.
        splits:      Subdirectory names to process (default: train + valid).
        img_size:    Output crop size in pixels (square).

    Returns:
        X           (N, img_size, img_size, 3) float32 ndarray, values in [0,1]
        y           (N,) int32 ndarray of class indices
        class_names list[str] of class name strings (index matches y values)
    """
    class_names = _parse_data_yaml(dataset_dir)
    crops, labels = [], []
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for split in splits:
        imgs_dir   = os.path.join(dataset_dir, split, 'images')
        labels_dir = os.path.join(dataset_dir, split, 'labels')

        if not os.path.isdir(imgs_dir):
            print(f'[dataset] skipping split "{split}": images/ dir not found at {imgs_dir}')
            continue

        for fname in sorted(os.listdir(imgs_dir)):
            if os.path.splitext(fname)[1].lower() not in img_exts:
                continue

            img_path   = os.path.join(imgs_dir, fname)
            label_path = os.path.join(labels_dir,
                                      os.path.splitext(fname)[0] + '.txt')

            if not os.path.isfile(label_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
                    crop = _crop_box(img, cx_n, cy_n, w_n, h_n, img_size)
                    if crop is not None:
                        crops.append(crop)
                        labels.append(cls_id)

    if not crops:
        raise RuntimeError(
            f'No crops extracted from {dataset_dir}. '
            'Check that images/ and labels/ directories exist within each split.'
        )

    X = np.stack(crops).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    # Class distribution summary
    unique, counts = np.unique(y, return_counts=True)
    dist = {class_names[i]: int(c) for i, c in zip(unique, counts)}
    print(f'[dataset] {len(X)} crops | {len(class_names)} classes')
    print(f'[dataset] class distribution: {dist}')

    return X, y, class_names


def make_tf_datasets(X, y, num_classes: int,
                     val_split: float = 0.10,
                     test_split: float = 0.10,
                     batch_size: int = 32,
                     augment: bool = True,
                     seed: int = 42):
    """
    Shuffle, split, and build batched tf.data.Dataset pipelines.

    Args:
        X, y:        Arrays from load_yolo_crops().
        num_classes: Number of output classes.
        val_split:   Fraction of data for validation (default 10%).
        test_split:  Fraction of data for testing   (default 10%).
        batch_size:  Batch size for all three splits.
        augment:     Apply random flip/rotation/brightness to training data.
        seed:        Random seed for reproducibility.

    Returns:
        train_ds, val_ds, test_ds  — batched and prefetched tf.data.Dataset
        (X_test, y_test)           — raw numpy arrays for sklearn metrics
    """
    import tensorflow as tf

    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    n_test = int(n * test_split)
    n_val  = int(n * val_split)

    X_test,  y_test  = X[:n_test],               y[:n_test]
    X_val,   y_val   = X[n_test:n_test + n_val],  y[n_test:n_test + n_val]
    X_train, y_train = X[n_test + n_val:],         y[n_test + n_val:]

    print(f'[dataset] split  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}')

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  num_classes)

    AUTOTUNE = tf.data.AUTOTUNE

    aug_pipeline = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomBrightness(0.15),
    ], name='augmentation')

    def augment_fn(x, y_label):
        return aug_pipeline(x, training=True), y_label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
        .shuffle(len(X_train), seed=seed)
        .batch(batch_size)
    )
    if augment:
        train_ds = train_ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test_oh))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, (X_test, y_test)


def visualise_samples(X, y, class_names: list, n: int = 12,
                      output_path: str = None) -> None:
    """Display a grid of n randomly selected crop samples with class labels."""
    rng  = np.random.default_rng(0)
    idx  = rng.choice(len(X), size=min(n, len(X)), replace=False)
    cols = 4
    rows = (len(idx) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for ax, i in zip(axes, idx):
        ax.imshow(X[i])
        ax.set_title(class_names[y[i]], fontsize=9)
        ax.axis('off')
    for ax in axes[len(idx):]:
        ax.axis('off')

    plt.suptitle('B.1 — Sample Crops (YOLO bbox regions)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved ->', output_path)
