from __future__ import annotations

import base64
import cv2

from config import IMAGE_CONFIG, build_prediction_result

from .gradcam import generate_gradcam_heatmap, overlay_heatmap
from .image_model import load_image_model
from .image_utils import decode_image_bytes, preprocess_image_bytes


def predict_image_bytes(image_bytes: bytes, filename: str | None = None) -> dict:
    loaded_model = load_image_model()
    batch, meta = preprocess_image_bytes(image_bytes, mode=loaded_model.preprocessing)
    raw_prediction = loaded_model.model.predict(batch, verbose=0)[0]

    # Generate Grad-CAM Attention Heatmap
    heatmap_b64: str | None = None
    try:
        heatmap = generate_gradcam_heatmap(loaded_model.model, batch, pred_index=1)
        original_bgr = decode_image_bytes(image_bytes)
        overlay_rgb = overlay_heatmap(heatmap, original_bgr)
        success, encoded = cv2.imencode(".png", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        if success:
            heatmap_b64 = base64.b64encode(encoded).decode("utf-8")
    except Exception as exc:
        print(f"[WARN] Failed to generate Grad-CAM heatmap: {exc}")

    return build_prediction_result(
        modality="image",
        raw_prediction=raw_prediction,
        threshold=IMAGE_CONFIG.threshold,
        model_source=loaded_model.source_path,
        extra={
            "filename": filename,
            "input_shape": list(batch.shape[1:]),
            "preprocessing": loaded_model.preprocessing,
            "legacy_binary_head": loaded_model.legacy_binary_head,
            "face_detected": meta.get("face_detected", False),
            "face_bbox": meta.get("bbox"),
            "heatmap_b64": heatmap_b64,
        },
    )
