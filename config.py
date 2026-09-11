from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent

REAL_INDEX = 0
FAKE_INDEX = 1
LABEL_MAP = {
    REAL_INDEX: "REAL",
    FAKE_INDEX: "FAKE",
}


@dataclass(frozen=True)
class ImageConfig:
    width: int = int(os.getenv("IMAGE_WIDTH", "224"))
    height: int = int(os.getenv("IMAGE_HEIGHT", "224"))
    threshold: float = float(os.getenv("IMAGE_THRESHOLD", "0.5"))
    compliant_model_name: str = os.getenv("IMAGE_COMPLIANT_MODEL", "efficientnet_deepfake.keras")
    legacy_model_name: str = os.getenv("IMAGE_LEGACY_MODEL", "deepfake_cnn.keras")

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def compliant_model_path(self) -> Path:
        return ROOT_DIR / "image_model" / self.compliant_model_name

    @property
    def legacy_model_path(self) -> Path:
        return ROOT_DIR / "image_model" / self.legacy_model_name


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    target_samples: int = int(os.getenv("AUDIO_TARGET_SAMPLES", "32000"))
    num_mfcc: int = int(os.getenv("AUDIO_NUM_MFCC", "40"))
    max_frames: int = int(os.getenv("AUDIO_MAX_FRAMES", "100"))
    threshold: float = float(os.getenv("AUDIO_THRESHOLD", "0.5"))
    compliant_model_name: str = os.getenv("AUDIO_COMPLIANT_MODEL", "audio_classifier.keras")
    legacy_model_name: str = os.getenv("AUDIO_LEGACY_MODEL", "audio_deepfake_model.keras")
    legacy_weights_name: str = os.getenv("AUDIO_LEGACY_WEIGHTS", "audio.weights.h5")

    @property
    def compliant_model_path(self) -> Path:
        return ROOT_DIR / "audio_model" / self.compliant_model_name

    @property
    def legacy_model_path(self) -> Path:
        return ROOT_DIR / "audio_model" / self.legacy_model_name

    @property
    def legacy_weights_path(self) -> Path:
        return ROOT_DIR / "audio_model" / self.legacy_weights_name


@dataclass(frozen=True)
class VideoConfig:
    frame_count: int = int(os.getenv("VIDEO_FRAME_COUNT", "16"))
    frame_stride: int = int(os.getenv("VIDEO_FRAME_STRIDE", "2"))
    threshold: float = float(os.getenv("VIDEO_THRESHOLD", "0.5"))


IMAGE_CONFIG = ImageConfig()
AUDIO_CONFIG = AudioConfig()
VIDEO_CONFIG = VideoConfig()


def normalize_probabilities(raw_prediction: Any) -> np.ndarray:
    probabilities = np.asarray(raw_prediction, dtype=np.float32).reshape(-1)
    if probabilities.size == 0:
        raise ValueError("Model returned an empty prediction.")

    if probabilities.size == 1:
        fake_probability = float(probabilities[0])
        probabilities = np.array([1.0 - fake_probability, fake_probability], dtype=np.float32)
    elif probabilities.size != 2:
        raise ValueError(f"Expected 1 or 2 output values, received {probabilities.size}.")

    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("Prediction probabilities sum to zero.")

    return (probabilities / total).astype(np.float32)


def build_prediction_result(
    modality: str,
    raw_prediction: Any,
    threshold: float,
    model_source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probabilities = normalize_probabilities(raw_prediction)
    fake_probability = float(probabilities[FAKE_INDEX])
    class_index = FAKE_INDEX if fake_probability >= threshold else REAL_INDEX
    confidence = float(probabilities[class_index])

    print(
        f"[{modality}] raw_probabilities={probabilities.tolist()} "
        f"class_index={class_index} label={LABEL_MAP[class_index]} confidence={confidence:.6f}"
    )

    payload: dict[str, Any] = {
        "modality": modality,
        "label_mapping": LABEL_MAP,
        "threshold": threshold,
        "raw_probabilities": probabilities.tolist(),
        "class_index": class_index,
        "label": LABEL_MAP[class_index],
        "confidence": confidence,
        "is_deepfake": bool(class_index == FAKE_INDEX),
        "probability_fake": fake_probability,
        "model_source": model_source,
    }

    if extra:
        payload.update(extra)

    return payload
