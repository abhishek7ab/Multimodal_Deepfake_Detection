from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from config import IMAGE_CONFIG


_FACE_CASCADE: cv2.CascadeClassifier | None = None


def get_face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode the uploaded image.")
    return image_bgr


def detect_and_crop_face(image_bgr: np.ndarray, margin: float = 0.25) -> tuple[np.ndarray, dict]:
    """
    Detects the primary face in the image and crops it with a margin.
    If no face is detected, returns a center square crop.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade = get_face_cascade()

    # Search the upper 70% of portrait/full-body images to eliminate false positives on clothing, pads, or knees
    upper_limit = int(h * 0.70) if h > w else h
    faces = cascade.detectMultiScale(
        gray[:upper_limit, :],
        scaleFactor=1.08,
        minNeighbors=3,
        minSize=(24, 24),
    )

    if len(faces) == 0:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(24, 24),
        )

    if len(faces) > 0:
        best_face = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        x, y, fw, fh = (int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3]))
        mx = int(fw * margin)
        my = int(fh * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + fw + mx)
        y2 = min(h, y + fh + my)

        cropped = image_bgr[y1:y2, x1:x2]
        meta = {
            "face_detected": True,
            "bbox": [x, y, fw, fh],
            "crop_coords": [x1, y1, x2, y2],
        }
        return cropped, meta

    side = min(h, w)
    cy, cx = h // 2, w // 2
    half = side // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, x1 + side), min(h, y1 + side)
    cropped = image_bgr[y1:y2, x1:x2]
    meta = {
        "face_detected": False,
        "bbox": None,
        "crop_coords": [x1, y1, x2, y2],
    }
    return cropped, meta


def prepare_rgb_image(image_bgr: np.ndarray, crop_face: bool = True) -> tuple[np.ndarray, dict]:
    if crop_face:
        cropped, meta = detect_and_crop_face(image_bgr)
    else:
        cropped = image_bgr
        meta = {"face_detected": False, "bbox": None, "crop_coords": None}

    image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, IMAGE_CONFIG.size, interpolation=cv2.INTER_AREA)
    return image_rgb.astype(np.float32), meta


def preprocess_image_array(
    image_bgr: np.ndarray,
    mode: str = "efficientnet",
    crop_face: bool = True,
) -> tuple[np.ndarray, dict]:
    image_rgb, meta = prepare_rgb_image(image_bgr, crop_face=crop_face)

    if mode == "efficientnet":
        processed = tf.keras.applications.efficientnet.preprocess_input(image_rgb)
    elif mode in ("legacy_rgb", "legacy_rgb_normalized"):
        processed = image_rgb / 255.0
    else:
        raise ValueError(f"Unknown image preprocessing mode: {mode}")

    batch = np.expand_dims(processed, axis=0).astype(np.float32)
    return batch, meta


def preprocess_image_bytes(
    image_bytes: bytes,
    mode: str = "efficientnet",
    crop_face: bool = True,
) -> tuple[np.ndarray, dict]:
    image_bgr = decode_image_bytes(image_bytes)
    return preprocess_image_array(image_bgr, mode=mode, crop_face=crop_face)
