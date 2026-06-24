"""Image preprocessing utilities.

The same preprocessing must be applied at training time and at inference time,
otherwise predictions silently degrade. Keep this module as the single source
of truth for how a raw image becomes a model-ready tensor.
"""

import numpy as np
import cv2
# load_img / img_to_array live in keras.utils (works in Keras 2.9+ and Keras 3).
from tensorflow.keras.utils import img_to_array, load_img


def preprocess_array(image, image_size):
    """Resize and scale a single RGB image array to [0, 1].

    Args:
        image: HxWx3 uint8/float RGB numpy array.
        image_size: (height, width) target size.
    Returns:
        Float32 array of shape (height, width, 3) scaled to [0, 1].
    """
    image = cv2.resize(image, (image_size[1], image_size[0]))
    image = image.astype("float32") / 255.0
    return image


def load_and_preprocess(path, image_size):
    """Load an image file from disk and return a model-ready array."""
    img = load_img(path, target_size=(image_size[0], image_size[1]))
    arr = img_to_array(img).astype("float32") / 255.0
    return arr


def make_batch(image_array):
    """Add the batch dimension expected by Keras models."""
    return np.expand_dims(image_array, axis=0)
