"""Model definitions: a custom CNN baseline and transfer-learning backbones.

Two philosophies live here:
  * `build_custom_cnn`   — a from-scratch CNN to establish a baseline.
  * `build_transfer_model` — MobileNetV2 / EfficientNetB0 pretrained on
    ImageNet, which usually wins on small-to-medium datasets because the
    backbone already knows generic visual features (edges, textures, shapes).
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

NUM_CLASSES = 10


def build_custom_cnn(input_shape, dropout=0.3, dense_units=256):
    """A compact VGG-style CNN trained from scratch (baseline)."""
    model = models.Sequential(name="custom_cnn")
    model.add(layers.Input(shape=input_shape))

    for filters in (32, 64, 128, 128):
        model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    return model


def _build_backbone(architecture, input_shape):
    """Instantiate a frozen ImageNet backbone (without its classifier head)."""
    if architecture == "mobilenetv2":
        base = MobileNetV2(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif architecture == "efficientnetb0":
        base = EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown transfer architecture: {architecture}")
    return base


def build_transfer_model(architecture, input_shape, dropout=0.3,
                         dense_units=256, freeze_base=True):
    """Build a transfer-learning model on top of a pretrained backbone."""
    base = _build_backbone(architecture, input_shape)
    base.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"{architecture}_transfer")
    model._backbone = base  # convenience handle for fine-tuning
    return model


def unfreeze_for_fine_tuning(model, fine_tune_at=100):
    """Unfreeze the upper layers of the backbone for stage-2 fine-tuning."""
    base = getattr(model, "_backbone", None)
    if base is None:
        # Fallback: find the nested Model layer (the backbone).
        base = next(l for l in model.layers if isinstance(l, models.Model))
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    return model


def build_model(config):
    """Factory that returns the model selected in config.yaml."""
    arch = config["model"]["architecture"]
    image_size = config["data"]["image_size"]
    input_shape = (image_size[0], image_size[1], config["data"]["channels"])
    m_cfg = config["model"]

    if arch == "custom_cnn":
        return build_custom_cnn(
            input_shape, dropout=m_cfg["dropout"], dense_units=m_cfg["dense_units"]
        )
    return build_transfer_model(
        arch,
        input_shape,
        dropout=m_cfg["dropout"],
        dense_units=m_cfg["dense_units"],
        freeze_base=m_cfg["freeze_base"],
    )
