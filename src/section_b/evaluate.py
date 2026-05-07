"""
evaluate.py  —  Section B: Model Evaluation & Visualisation
------------------------------------------------------------
Classification metrics, confusion matrix heatmap, prediction grid,
and multi-config comparison bar chart.

Public API
----------
evaluate_model(model, X_test, y_test, class_names)  -> dict
plot_confusion_matrix(y_true, y_pred, class_names, output_path)
plot_prediction_grid(model, X_test, y_test, class_names, n, output_path)
compare_configs(results, output_path)               -> None
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   class_names: list) -> dict:
    """
    Compute accuracy, weighted precision, recall, F1 and a full per-class report.

    Args:
        model:       Trained tf.keras.Model.
        X_test:      Test images  (N, H, W, 3) float32.
        y_test:      Integer class labels (N,).
        class_names: List of class name strings.

    Returns:
        dict with keys: accuracy, precision, recall, f1, report (str), y_pred (ndarray)
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report,
    )

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted',
                                     zero_division=0),
        'recall':    recall_score(y_test, y_pred,    average='weighted',
                                  zero_division=0),
        'f1':        f1_score(y_test, y_pred,         average='weighted',
                              zero_division=0),
        'report':    classification_report(y_test, y_pred,
                                           target_names=class_names,
                                           zero_division=0),
        'y_pred':    y_pred,
    }

    print(f"\nTest accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision (W)  : {metrics['precision']:.4f}")
    print(f"Recall    (W)  : {metrics['recall']:.4f}")
    print(f"F1        (W)  : {metrics['f1']:.4f}")
    print(f"\nPer-class report:\n{metrics['report']}")

    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names: list,
                          output_path: str = None):
    """Plot a normalised seaborn confusion-matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    try:
        import seaborn as sns
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'seaborn'],
                       check=True)
        import seaborn as sns

    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    size = max(6, len(class_names) * 1.4)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix (Normalised)', fontsize=13, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    if output_path:
        print('Saved ->', output_path)


# ---------------------------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------------------------

def plot_prediction_grid(model, X_test: np.ndarray, y_test: np.ndarray,
                         class_names: list, n: int = 12,
                         output_path: str = None):
    """
    Display n test crops with actual vs predicted labels.
    Green title = correct prediction, red title = incorrect.
    """
    rng   = np.random.default_rng(0)
    idx   = rng.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    preds = np.argmax(model.predict(X_test[idx], verbose=0), axis=1)

    cols  = 4
    rows  = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    axes = np.array(axes).flatten()

    for ax, i, pred in zip(axes, idx, preds):
        ax.imshow(X_test[i])
        true_name = class_names[y_test[i]]
        pred_name = class_names[pred]
        colour    = 'green' if pred == y_test[i] else 'red'
        ax.set_title(f'True: {true_name}\nPred: {pred_name}',
                     fontsize=8, color=colour)
        ax.axis('off')
    for ax in axes[len(idx):]:
        ax.axis('off')

    plt.suptitle('B.4 — Predictions  (green = correct,  red = wrong)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    if output_path:
        print('Saved ->', output_path)


# ---------------------------------------------------------------------------
# Multi-config comparison
# ---------------------------------------------------------------------------

def compare_configs(results: dict, output_path: str = None):
    """
    Grouped bar chart comparing accuracy, precision, recall, F1 across configs.

    Args:
        results: {config_label_str: metrics_dict, ...}
                 metrics_dict must have keys: accuracy, precision, recall, f1
    """
    labels  = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colours = ['steelblue', 'darkorange', 'seagreen', 'orchid']
    x       = np.arange(len(labels))
    width   = 0.18

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 3.5), 6))

    for i, (metric, colour) in enumerate(zip(metrics, colours)):
        vals = [results[lbl][metric] for lbl in labels]
        bars = ax.bar(x + i * width, vals, width,
                      label=metric.capitalize(), color=colour, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel('Score')
    ax.set_title('B.4 — Config Comparison: Accuracy / Precision / Recall / F1',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    if output_path:
        print('Saved ->', output_path)
