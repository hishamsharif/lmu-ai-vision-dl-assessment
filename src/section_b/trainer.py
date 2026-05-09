"""
trainer.py  —  Section B: Hyperparameter Configurations & Training Loop
------------------------------------------------------------------------
Defines three distinct training configurations and provides a
compile_and_train() helper used by Section B notebook cells.

Configurations
--------------
Config 1 — SGD  LR=0.01  dropout=0.30  epochs=30
    Baseline: momentum SGD with a relatively high learning rate.
    Explores fast convergence vs potential instability trade-off.

Config 2 — Adam  LR=0.0001  dropout=0.50  epochs=30
    Conservative Adam: very low learning rate.
    Stable but may under-converge within 30 epochs.

Config 3 — Adam  LR=0.001  dropout=0.50  epochs=50  patience=7
    Standard Adam with early stopping.
    Expected best performer: adaptive LR + regularisation.

Public API
----------
CONFIGS                   dict of config dicts
compile_and_train(...)    -> tf.keras.callbacks.History
plot_training_curves(...) -> None
"""

import os
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Hyperparameter registry
# ---------------------------------------------------------------------------

CONFIGS = {
    'config1': {
        'label':       'Config 1 — SGD LR=0.01',
        'optimizer':   'sgd',
        'lr':          0.01,
        'dropout':     0.30,
        'epochs':      30,
        'patience':    None,
        'description': (
            'Stochastic Gradient Descent with momentum=0.9 and LR=0.01. '
            'High learning rate drives fast early convergence but can overshoot '
            'narrow minima in later epochs.'
        ),
    },
    'config2': {
        'label':       'Config 2 — Adam LR=0.0001',
        'optimizer':   'adam',
        'lr':          0.0001,
        'dropout':     0.50,
        'epochs':      30,
        'description': (
            'Adam optimiser with a conservative LR=0.0001. '
            'Adaptive moment estimation provides stable updates; low LR '
            'reduces risk of divergence but may need more epochs to plateau.'
        ),
        'patience':    None,
    },
    'config3': {
        'label':       'Config 3 — Adam LR=0.001 + EarlyStopping',
        'optimizer':   'adam',
        'lr':          0.001,
        'dropout':     0.50,
        'epochs':      50,
        'patience':    7,
        'description': (
            'Adam with the standard Keras default LR=0.001 and early stopping '
            '(patience=7 on val_accuracy).  Higher dropout (0.50) regularises '
            'the dense head.  Expected to deliver the best generalisation.'
        ),
    },
}


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def compile_and_train(model, train_ds, val_ds, config: dict,
                      output_dir: str = None):
    """
    Compile model with the given config settings and run training.

    A fresh copy of the model's weights is NOT made automatically — rebuild
    the model (build_model()) before each config run to avoid weight leakage.

    Args:
        model:      A tf.keras.Model from model.build_model().
        train_ds:   Training tf.data.Dataset (batched, augmented).
        val_ds:     Validation tf.data.Dataset (batched).
        config:     One entry from CONFIGS (dict with keys: optimizer, lr,
                    epochs, patience, label).
        output_dir: If provided, saves best checkpoint as best_model.keras.

    Returns:
        history: tf.keras.callbacks.History object.
    """
    import tensorflow as tf

    opt_name = config['optimizer'].lower()
    lr       = config['lr']

    if opt_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, 'best_model.keras')
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
            )
        )

    if config.get('patience'):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=config['patience'],
                restore_best_weights=True,
                verbose=1,
            )
        )

    print(f"\n{'='*60}")
    print(f"Training: {config['label']}")
    print(f"  optimizer : {opt_name}  lr={lr}")
    print(f"  epochs    : {config['epochs']}  patience={config.get('patience','—')}")
    print(f"{'='*60}\n")

    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_training_curves(history, config_label: str,
                         output_path: str = None) -> None:
    """Plot accuracy and loss curves for a single training run side-by-side."""
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Curves — {config_label}', fontsize=13, fontweight='bold')

    ax1.plot(epochs, acc,     'b-o', ms=4, label='Train Accuracy')
    ax1.plot(epochs, val_acc, 'r-s', ms=4, label='Val Accuracy')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.plot(epochs, loss,     'b-o', ms=4, label='Train Loss')
    ax2.plot(epochs, val_loss, 'r-s', ms=4, label='Val Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    if output_path:
        print('Saved ->', output_path)
