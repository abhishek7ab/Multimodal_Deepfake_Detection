from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import AUDIO_CONFIG


@dataclass(frozen=True)
class LoadedAudioModel:
    model: tf.keras.Model
    source_path: str
    preprocessing: str
    legacy_binary_head: bool


def _wrap_binary_model(base_model: tf.keras.Model) -> tf.keras.Model:
    input_shape = tuple(base_model.input_shape[1:])
    inputs = tf.keras.Input(shape=input_shape, name="legacy_input")
    base_outputs = base_model(inputs)
    # The legacy model has a single sigmoid output representing the FAKE
    # probability. In canonical [REAL, FAKE] format:
    # Index 0 is REAL: 1.0 - tensor
    # Index 1 is FAKE: tensor
    outputs = tf.keras.layers.Lambda(
        lambda tensor: tf.concat([1.0 - tensor, tensor], axis=-1),
        name="binary_to_two_class_probabilities",
    )(base_outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{base_model.name}_wrapped")


def build_audio_model(input_shape: tuple[int, int, int] | None = None, num_classes: int = 2) -> tf.keras.Model:
    input_shape = input_shape or (AUDIO_CONFIG.num_mfcc, AUDIO_CONFIG.max_frames, 1)
    inputs = tf.keras.Input(shape=input_shape, name="mfcc")
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="audio_deepfake_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_legacy_audio_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(128, 63, 1), name="legacy_mel_spectrogram")
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="legacy_audio_deepfake_classifier")


def compute_audio_class_weights(labels: Sequence[int]) -> dict[int, float]:
    labels_array = np.asarray(labels, dtype=np.int32)
    classes = np.unique(labels_array)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels_array)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def _load_keras_model(model_path: Path) -> tf.keras.Model:
    if model_path.exists() and model_path.stat().st_size < 10000:
        try:
            import subprocess

            print(f"[Streamlit Cloud LFS] Auto-pulling LFS for {model_path}...")
            subprocess.run(["git", "lfs", "pull"], check=False, timeout=120)
        except Exception as exc:
            print(f"[WARN LFS] Could not auto-pull git-lfs: {exc}")
    return tf.keras.models.load_model(model_path, compile=False)


@lru_cache(maxsize=1)
def load_audio_model() -> LoadedAudioModel:
    if AUDIO_CONFIG.compliant_model_path.exists():
        model = _load_keras_model(AUDIO_CONFIG.compliant_model_path)
        output_units = int(model.output_shape[-1])

        if output_units == 2:
            return LoadedAudioModel(
                model=model,
                source_path=str(AUDIO_CONFIG.compliant_model_path),
                preprocessing="mfcc_40",
                legacy_binary_head=False,
            )

        if output_units == 1:
            wrapped = _wrap_binary_model(model)
            return LoadedAudioModel(
                model=wrapped,
                source_path=str(AUDIO_CONFIG.compliant_model_path),
                preprocessing="legacy_log_mel",
                legacy_binary_head=True,
            )

        raise ValueError(f"Unsupported audio model output shape: {model.output_shape}")

    if AUDIO_CONFIG.legacy_model_path.exists():
        model = _load_keras_model(AUDIO_CONFIG.legacy_model_path)
        output_units = int(model.output_shape[-1])

        if output_units == 2:
            return LoadedAudioModel(
                model=model,
                source_path=str(AUDIO_CONFIG.legacy_model_path),
                preprocessing="mfcc_40",
                legacy_binary_head=False,
            )

        if output_units == 1:
            wrapped = _wrap_binary_model(model)
            return LoadedAudioModel(
                model=wrapped,
                source_path=str(AUDIO_CONFIG.legacy_model_path),
                preprocessing="legacy_log_mel",
                legacy_binary_head=True,
            )

        raise ValueError(f"Unsupported audio model output shape: {model.output_shape}")

    if AUDIO_CONFIG.legacy_weights_path.exists():
        legacy_model = build_legacy_audio_model()
        legacy_model.load_weights(AUDIO_CONFIG.legacy_weights_path)
        wrapped = _wrap_binary_model(legacy_model)
        return LoadedAudioModel(
            model=wrapped,
            source_path=str(AUDIO_CONFIG.legacy_weights_path),
            preprocessing="legacy_log_mel",
            legacy_binary_head=True,
        )

    checked_paths = ", ".join(
        str(path)
        for path in [
            AUDIO_CONFIG.compliant_model_path,
            AUDIO_CONFIG.legacy_model_path,
            AUDIO_CONFIG.legacy_weights_path,
        ]
    )
    raise FileNotFoundError(f"No audio model found. Checked: {checked_paths}")
