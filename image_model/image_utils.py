from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from config import IMAGE_CONFIG


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode the uploaded image.")
    return image_bgr


def prepare_rgb_image(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, IMAGE_CONFIG.size, interpolation=cv2.INTER_AREA)
    return image_rgb.astype(np.float32)


def preprocess_image_array(image_bgr: np.ndarray, mode: str = "efficientnet") -> np.ndarray:
    image_rgb = prepare_rgb_image(image_bgr)

    if mode == "efficientnet":
        # The compliant model expects the preprocessing convention used by
        # Keras EfficientNet.
        image_rgb = tf.keras.applications.efficientnet.preprocess_input(image_rgb)
    elif mode == "legacy_rgb":
        # The legacy deepfake_cnn.keras contains the EfficientNet backbone's
        # own Rescaling layer, so inference must receive RGB pixels in the
        # original 0..255 range. Dividing by 255 here would double-normalize
        # the input and change the distribution seen during training.
        pass
    else:
        raise ValueError(f"Unknown image preprocessing mode: {mode}")

    return np.expand_dims(image_rgb, axis=0).astype(np.float32)


def preprocess_image_bytes(image_bytes: bytes, mode: str = "efficientnet") -> np.ndarray:
    image_bgr = decode_image_bytes(image_bytes)
    return preprocess_image_array(image_bgr, mode=mode)
