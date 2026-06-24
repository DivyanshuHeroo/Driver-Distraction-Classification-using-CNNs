"""Data loading: build augmented train / validation `tf.data` pipelines.

Expects the Kaggle State Farm folder layout:

    data/imgs/train/
        c0/ img_1.jpg ...
        c1/ ...
        ...
        c9/ ...

Uses `image_dataset_from_directory` + Keras preprocessing layers rather than
the legacy `ImageDataGenerator`. This is the modern, efficient path and works
across TensorFlow 2.12–2.21 (both Keras 2 and Keras 3). Images are scaled to
[0, 1] so the pipeline matches `src/preprocess.py` used at inference time.
"""

import os

import tensorflow as tf
from tensorflow.keras import layers, Sequential

from src.utils import get_class_names

AUTOTUNE = tf.data.AUTOTUNE


def _build_augmenter(aug):
    """A Sequential of random-augmentation layers applied to training images.

    Augmentation only runs during training (the layers are no-ops at
    inference), which expands the effective dataset and curbs overfitting.
    """
    # Brightness factor is derived from the configured multiplicative range,
    # e.g. [0.8, 1.2] -> +/-0.2 jitter.
    br = aug.get("brightness_range", [1.0, 1.0])
    brightness_factor = max(0.0, (br[1] - br[0]) / 2.0)

    aug_layers = []
    if aug.get("horizontal_flip"):
        aug_layers.append(layers.RandomFlip("horizontal"))
    aug_layers += [
        # RandomRotation factor is a fraction of a full turn: deg / 360.
        layers.RandomRotation(aug["rotation_range"] / 360.0),
        layers.RandomTranslation(
            aug["height_shift_range"], aug["width_shift_range"]
        ),
        layers.RandomZoom(aug["zoom_range"]),
    ]
    if brightness_factor > 0:
        aug_layers.append(
            layers.RandomBrightness(factor=brightness_factor, value_range=(0.0, 1.0))
        )
    return Sequential(aug_layers, name="augmentation")


def build_datasets(config):
    """Create training and validation `tf.data.Dataset`s.

    Returns:
        (train_ds, val_ds) yielding (images in [0,1], int labels).
    """
    data_cfg = config["data"]
    image_size = tuple(data_cfg["image_size"])
    class_names = get_class_names(config)
    seed = data_cfg["seed"]

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config["paths"]["data_dir"],
        validation_split=data_cfg["validation_split"],
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=data_cfg["batch_size"],
        class_names=class_names,
        label_mode="int",
        shuffle=True,
    )
    # NOTE: shuffle must stay True (with the same seed) here. The seeded
    # shuffle is what makes the train/validation split disjoint and stratified;
    # passing shuffle=False would split on raw class-ordered files and leak.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config["paths"]["data_dir"],
        validation_split=data_cfg["validation_split"],
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=data_cfg["batch_size"],
        class_names=class_names,
        label_mode="int",
        shuffle=True,
    )

    # Scale pixels to [0, 1] to match inference-time preprocessing.
    rescale = lambda x, y: (x / 255.0, y)
    train_ds = train_ds.map(rescale, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(rescale, num_parallel_calls=AUTOTUNE)

    # Apply augmentation to the training set only.
    augmenter = _build_augmenter(config["augmentation"])
    train_ds = train_ds.map(
        lambda x, y: (augmenter(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)


def compute_class_weights(config):
    """Return balanced class weights from per-class file counts.

    weight_c = total_images / (num_classes * count_c)
    Larger weight for rarer classes, so training does not ignore them.
    """
    class_names = get_class_names(config)
    data_dir = config["paths"]["data_dir"]
    counts = []
    for c in class_names:
        folder = os.path.join(data_dir, c)
        counts.append(len(os.listdir(folder)) if os.path.isdir(folder) else 0)

    total = sum(counts)
    n = len(class_names)
    weights = {}
    for i, count in enumerate(counts):
        weights[i] = total / (n * count) if count > 0 else 1.0
    return weights
