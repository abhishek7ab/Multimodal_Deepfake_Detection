from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import IMAGE_CONFIG


_FACE_CASCADE: cv2.CascadeClassifier | None = None


def get_face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        alt2_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        default_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade_path = alt2_path if Path(alt2_path).exists() else default_path
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

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(28, 28),
    )

    if len(faces) == 0:
        default_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        def_cascade = cv2.CascadeClassifier(default_path)
        faces = def_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(28, 28),
        )

    if len(faces) > 0:
        # Score candidates to prefer upper-center head regions over sports gloves, jersey logos, or knee pads
        def _score_candidate(f):
            fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            y_norm = fy / max(h, 1)
            x_center_dist = abs((fx + fw / 2.0) - w / 2.0) / max(w, 1)
            # Heavy penalty for detections located in lower half or extreme borders
            pos_penalty = (y_norm * 2.8) + (x_center_dist * 0.8)
            area_score = (fw * fh) / max(w * h, 1)
            return area_score - (pos_penalty * 0.15)

        best_face = max(faces, key=_score_candidate)
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
