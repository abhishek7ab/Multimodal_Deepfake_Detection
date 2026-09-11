from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import IMAGE_CONFIG


@dataclass(frozen=True)
class LoadedImageModel:
    model: tf.keras.Model
    source_path: str
    preprocessing: str
    legacy_binary_head: bool


def _wrap_binary_model(base_model: tf.keras.Model) -> tf.keras.Model:
    input_shape = tuple(base_model.input_shape[1:])
    inputs = tf.keras.Input(shape=input_shape, name="legacy_input")
    base_output = base_model(inputs)

    # The legacy model has a single sigmoid output representing the FAKE
    # probability. In canonical [REAL, FAKE] format:
    # Index 0 is REAL: 1.0 - tensor
    # Index 1 is FAKE: tensor
    outputs = tf.keras.layers.Lambda(
        lambda tensor: tf.concat([1.0 - tensor, tensor], axis=-1),
        name="binary_to_two_class_probabilities",
    )(base_output)
    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{base_model.name}_wrapped",
    )


def create_image_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="image_augmentation",
    )


def build_image_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    backbone_trainable: bool = False,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = create_image_augmentation()(inputs)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = backbone_trainable
    x = backbone(x, training=backbone_trainable)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_image_deepfake")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_image_class_weights(labels: Sequence[int]) -> dict[int, float]:
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
def load_image_model() -> LoadedImageModel:
    candidate_paths = [
        IMAGE_CONFIG.compliant_model_path,
        IMAGE_CONFIG.legacy_model_path,
    ]

    for model_path in candidate_paths:
        if not model_path.exists():
            continue

        model = _load_keras_model(model_path)
        output_units = int(model.output_shape[-1])

        if output_units == 2:
            return LoadedImageModel(
                model=model,
                source_path=str(model_path),
                preprocessing="efficientnet",
                legacy_binary_head=False,
            )

        if output_units == 1:
            wrapped = _wrap_binary_model(model)
            return LoadedImageModel(
                model=wrapped,
                source_path=str(model_path),
                preprocessing="legacy_rgb",
                legacy_binary_head=True,
            )

        raise ValueError(f"Unsupported image model output shape: {model.output_shape}")

    checked_paths = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"No image model found. Checked: {checked_paths}")
