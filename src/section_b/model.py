"""
model.py  —  Section B: MobileNetV2 Transfer-Learning Classifier
-----------------------------------------------------------------
Architecture:
  MobileNetV2 (ImageNet weights, all layers frozen)
    → GlobalAveragePooling2D
    → Dense(128, activation='relu')
    → Dropout(dropout)
    → Dense(num_classes, activation='softmax')

Public API
----------
build_model(num_classes, dropout=0.5, img_size=224) -> tf.keras.Model
plot_architecture(model, output_path=None)          -> str (saved png path)
"""


def build_model(num_classes: int, dropout: float = 0.5,
                img_size: int = 224):
    """
    Build a MobileNetV2-based PPE classification model.

    The MobileNetV2 backbone is loaded with ImageNet weights and kept
    fully frozen (feature extraction mode).  Only the custom classification
    head is trained.

    Args:
        num_classes: Number of PPE categories.
        dropout:     Dropout rate after the first Dense layer.
        img_size:    Input spatial resolution (square).

    Returns:
        tf.keras.Model (not yet compiled — call model.compile() separately).
    """
    import tensorflow as tf

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False          # freeze all backbone weights

    inputs  = tf.keras.Input(shape=(img_size, img_size, 3), name='input')
    x       = base_model(inputs, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x       = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(x)
    x       = tf.keras.layers.Dropout(dropout, name='dropout')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    name='output')(x)

    model = tf.keras.Model(inputs, outputs, name='PPE_MobileNetV2')
    return model


def plot_architecture(model, output_path: str = None) -> str:
    """
    Render and display a plot_model diagram of the model.

    Args:
        model:       A tf.keras.Model built with build_model().
        output_path: Where to save the PNG.  Defaults to /tmp/model_arch.png.

    Returns:
        Absolute path of the saved PNG.
    """
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    if output_path is None:
        output_path = '/tmp/model_arch.png'

    tf.keras.utils.plot_model(
        model,
        to_file=output_path,
        show_shapes=True,
        show_layer_names=True,
        dpi=96,
    )

    img = mpimg.imread(output_path)
    fig, ax = plt.subplots(figsize=(10, max(8, img.shape[0] / 96)))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Model Architecture — PPE_MobileNetV2', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return output_path
