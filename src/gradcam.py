"""Grad-CAM explainability.

Grad-CAM (Gradient-weighted Class Activation Mapping) answers the question
"which pixels made the model choose this class?". It does so by:
  1. Taking the gradients of the predicted class score w.r.t. the feature
     maps of the last convolutional layer.
  2. Global-average-pooling those gradients to get a weight per feature map.
  3. Weighting and summing the feature maps -> a coarse heatmap that is then
     upsampled and overlaid on the original image.

A heatmap focused on the driver's hands/phone is a sign the model is looking
at the right evidence rather than at spurious background cues.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models


def _output_rank(layer):
    """Best-effort rank of a layer's output (works in Keras 2 and Keras 3)."""
    try:
        return len(layer.output.shape)
    except Exception:
        try:
            return len(layer.output_shape)
        except Exception:
            return None


def find_last_conv_layer(model):
    """Return the name of the last convolutional layer, descending into backbones.

    Prefers an actual Conv layer (rank-4 output); falls back to the last
    rank-4 layer if no Conv is found by name.
    """
    last_rank4 = None
    for layer in reversed(model.layers):
        # Transfer models nest the backbone as a sub-Model; recurse into it.
        if isinstance(layer, models.Model):
            inner = find_last_conv_layer(layer)
            if inner:
                return inner
        if _output_rank(layer) == 4:
            if "conv" in layer.__class__.__name__.lower():
                return layer.name
            if last_rank4 is None:
                last_rank4 = layer.name
    return last_rank4


def _build_grad_model(model, last_conv_layer_name):
    """Build a model mapping inputs to (last-conv activations, predictions).

    The graph is reconstructed functionally on a fresh input so it works even
    for a model freshly loaded from disk (whose `.output` is not yet defined in
    Keras 3). Handles both flat/Sequential models and transfer models where the
    conv layer lives inside a nested backbone sub-model.
    """
    from tensorflow.keras import Input, layers as klayers

    # Is the conv layer nested inside a backbone sub-model?
    nested_base = None
    for layer in model.layers:
        if isinstance(layer, models.Model) and any(
            l.name == last_conv_layer_name for l in layer.layers
        ):
            nested_base = layer
            break

    input_shape = tuple(model.inputs[0].shape[1:])
    inp = Input(shape=input_shape)

    if nested_base is None:
        # Flat / Sequential model: chain layers, tapping the conv output.
        x = inp
        conv_out = None
        for layer in model.layers:
            if isinstance(layer, klayers.InputLayer):
                continue
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_out = x
        return models.Model(inp, [conv_out, x])

    # Transfer model: tap inside the backbone, then chain the remaining head.
    sub = models.Model(
        nested_base.inputs,
        [nested_base.get_layer(last_conv_layer_name).output, nested_base.output],
    )
    conv_out, base_out = sub(inp)
    x = base_out
    started = False
    for layer in model.layers:
        if layer is nested_base:
            started = True
            continue
        if not started or isinstance(layer, klayers.InputLayer):
            continue
        x = layer(x)
    return models.Model(inp, [conv_out, x])


def compute_gradcam(model, image_batch, last_conv_layer_name=None, pred_index=None):
    """Compute a normalized Grad-CAM heatmap for a single preprocessed image.

    Args:
        model: trained Keras model.
        image_batch: array of shape (1, H, W, 3) already scaled to [0, 1].
        last_conv_layer_name: optional explicit conv layer; auto-detected if None.
        pred_index: class to explain; defaults to the predicted class.
    Returns:
        (heatmap, pred_index, predictions)
        heatmap is a HxW float array in [0, 1].
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = _build_grad_model(model, last_conv_layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Keep only positive contributions and normalize to [0, 1].
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index, predictions.numpy()[0]


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """Overlay a Grad-CAM heatmap on the original RGB image.

    Args:
        original_image: HxWx3 uint8 RGB image (any size).
        heatmap: small float heatmap in [0, 1].
        alpha: overlay opacity.
    Returns:
        HxWx3 uint8 RGB image with the heatmap blended in.
    """
    h, w = original_image.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_image, 1 - alpha, colored, alpha, 0)
    return overlay
