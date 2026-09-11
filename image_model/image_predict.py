from __future__ import annotations

from config import IMAGE_CONFIG, build_prediction_result

from .image_model import load_image_model
from .image_utils import preprocess_image_bytes


def predict_image_bytes(image_bytes: bytes, filename: str | None = None) -> dict:
    loaded_model = load_image_model()
    batch = preprocess_image_bytes(image_bytes, mode=loaded_model.preprocessing)
    raw_prediction = loaded_model.model.predict(batch, verbose=0)[0]

    result = build_prediction_result(
        modality="image",
        raw_prediction=raw_prediction,
        threshold=IMAGE_CONFIG.threshold,
        model_source=loaded_model.source_path,
        extra={
            "filename": filename,
            "input_shape": list(batch.shape[1:]),
            "preprocessing": loaded_model.preprocessing,
            "legacy_binary_head": loaded_model.legacy_binary_head,
        },
    )

    # Keep compatibility with the Streamlit UI's result fields while using
    # the canonical REAL=0 / FAKE=1 mapping from config.py.
    result["probability_fake"] = float(result["raw_probabilities"][1])
    result["is_deepfake"] = result["label"] == "FAKE"

    return result
