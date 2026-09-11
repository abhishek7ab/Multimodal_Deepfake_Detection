from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


def find_last_conv_layer_name(model: tf.keras.Model) -> str | None:
    """Finds the name of the last 4D convolutional or activation layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if ("conv" in sub_layer.name.lower() or "activation" in sub_layer.name.lower()) and len(getattr(sub_layer, "output_shape", ())) == 4:
                    return sub_layer.name
        if "conv" in layer.name.lower() and len(getattr(layer, "output_shape", ())) == 4:
            return layer.name
    return None


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    pred_index: int = 1,
) -> np.ndarray:
    """
    Computes a Grad-CAM heatmap for the given image array.
    Falls back gracefully to high-frequency spatial activation if gradient computation is unsupported.
    """
    try:
        # Check if model has a nested backbone (e.g. efficientnet)
        backbone = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                backbone = layer
                break

        target_model = backbone if backbone is not None else model
        last_conv_name = find_last_conv_layer_name(target_model)

        if last_conv_name is not None:
            # Create a sub-model that maps inputs to last conv layer and predictions
            last_conv_layer = target_model.get_layer(last_conv_name)
            grad_model = tf.keras.Model(
                inputs=target_model.inputs,
                outputs=[last_conv_layer.output, target_model.outputs[0]],
            )

            with tf.GradientTape() as tape:
                inputs = tf.cast(img_array, tf.float32)
                conv_outputs, predictions = grad_model(inputs)
                if predictions.shape[-1] > 1:
                    loss = predictions[:, pred_index]
                else:
                    loss = predictions[:, 0]

            grads = tape.gradient(loss, conv_outputs)
            if grads is not None:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0.0) / (tf.reduce_max(heatmap) + 1e-10)
                heatmap_np = heatmap.numpy()
                if heatmap_np.ndim == 2 and not np.isnan(heatmap_np).any():
                    return heatmap_np.astype(np.float32)
    except Exception as exc:
        print(f"[WARN Grad-CAM] Gradient computation fell back to activation map: {exc}")

    # Fallback: compute spatial gradient magnitude / high-frequency texture artifact map
    sample = img_array[0]
    if sample.shape[-1] == 3:
        gray = cv2.cvtColor(np.uint8(np.clip((sample + 1.0) * 127.5 if sample.min() < 0 else sample * 255.0, 0, 255)), cv2.COLOR_RGB2GRAY)
    else:
        gray = np.uint8(sample)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    blurred = cv2.GaussianBlur(np.abs(laplacian), (21, 21), 0)
    norm = blurred / (np.max(blurred) + 1e-10)
    return norm.astype(np.float32)


def overlay_heatmap(
    heatmap: np.ndarray,
    original_bgr: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlays the normalized heatmap [0, 1] onto the original image.
    Returns RGB image ready for display.
    """
    h, w = original_bgr.shape[:2]
    resized_heatmap = cv2.resize(heatmap, (w, h))
    scaled_heatmap = np.uint8(255 * np.clip(resized_heatmap, 0, 1))

    color_heatmap = cv2.applyColorMap(scaled_heatmap, colormap)
    superimposed = cv2.addWeighted(color_heatmap, alpha, original_bgr, 1.0 - alpha, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    return superimposed_rgb
