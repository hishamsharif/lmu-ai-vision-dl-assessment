"""
hyperparams.py  —  Task 4: Hyperparameter Configurations
---------------------------------------------------------
Defines three distinct training configurations for EfficientDet-D0.
Each entry maps directly to fields patched into pipeline_template.config
by src/section_b/model.py:patch_pipeline_config().

Configuration design rationale
-------------------------------
Config 1 — Baseline
    Default EfficientDet-D0 cosine-decay schedule from the TF Model Zoo.
    Conservative learning rate, moderate batch size, no extra regularisation.
    Expected: stable convergence; may plateau and overfit after ~2500 steps.

Config 2 — Aggressive LR / Small Batch
    2x higher base LR with a smaller batch size produces noisier gradients
    that can help escape sharp local minima.  Shorter warmup means the
    high LR is applied earlier.
    Expected: faster initial loss drop; risk of instability (NaN loss).
    Mitigation: if NaN occurs, halve lr_base to 0.04.

Config 3 — Regularised / Long Run
    Lower LR, larger batch, higher dropout, longer training and warmup.
    Intended to improve generalisation at the cost of slower convergence.
    Expected: lower final val loss; requires more Colab GPU time.
    Note: Config 3 (6000 steps) may need a Colab Pro session or two runs.
"""

# Number of PPE classes: helmet (1), vest (2), goggles (3)
# Update this value if the verified dataset has a different class count.
NUM_CLASSES = 3

CONFIGS = {
    # ------------------------------------------------------------------
    "config1": {
        "description":      "Baseline — default EfficientDet-D0 cosine-decay LR",
        "lr_base":          0.04,
        "warmup_lr":        0.001,
        "warmup_steps":     1000,
        "num_steps":        3000,
        "batch_size":       8,
        "dropout_keep_prob": 0.8,
        "augmentation": [
            "random_horizontal_flip",
        ],
    },

    # ------------------------------------------------------------------
    "config2": {
        "description":      "Aggressive LR — 2x base LR, half batch, shorter warmup",
        "lr_base":          0.08,
        "warmup_lr":        0.002,
        "warmup_steps":     500,
        "num_steps":        3000,
        "batch_size":       4,
        "dropout_keep_prob": 0.8,
        "augmentation": [
            "random_horizontal_flip",
            "random_adjust_brightness",
        ],
    },

    # ------------------------------------------------------------------
    "config3": {
        "description":      "Regularised — low LR, high dropout, extended training",
        "lr_base":          0.008,
        "warmup_lr":        0.0001,
        "warmup_steps":     2000,
        "num_steps":        6000,
        "batch_size":       16,
        "dropout_keep_prob": 0.5,
        "augmentation": [
            "random_horizontal_flip",
            "random_adjust_brightness",
            "random_adjust_contrast",
        ],
    },
}
