from __future__ import annotations

import base64
import sys
from pathlib import Path

# Ensure repository root is on sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from config import IMAGE_CONFIG, build_prediction_result

try:
    from .gradcam import generate_gradcam_heatmap, overlay_heatmap
    from .image_model import load_image_model
    from .image_utils import decode_image_bytes, detect_and_crop_face, preprocess_image_bytes
except (ImportError, ValueError):
    # pyrefly: ignore [missing-import]
    from image_model.gradcam import generate_gradcam_heatmap, overlay_heatmap
    from image_model.image_model import load_image_model
    from image_model.image_utils import decode_image_bytes, detect_and_crop_face, preprocess_image_bytes



def predict_image_bytes(image_bytes: bytes, filename: str | None = None) -> dict:
    loaded_model = load_image_model()
    original_bgr = decode_image_bytes(image_bytes)

    # Multi-scale inspection:
    # 1. Tight face crop (margin=0.25) -> checks inner facial synthesis & micro-texture artifacts
    # 2. Context crop (margin=0.55) -> checks face-swap boundaries, jawline, hair & neckline seams
    c_tight, meta_tight = detect_and_crop_face(original_bgr, margin=0.25)
    c_context, meta_context = detect_and_crop_face(original_bgr, margin=0.55)

    def _prepare_tensor(crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMAGE_CONFIG.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        if loaded_model.preprocessing == "efficientnet":
            import tensorflow as tf
            return tf.keras.applications.efficientnet.preprocess_input(resized)
        return resized / 255.0

    batch_tight = np.expand_dims(_prepare_tensor(c_tight), axis=0)
    batch_context = np.expand_dims(_prepare_tensor(c_context), axis=0)
    batch_dual = np.concatenate([batch_tight, batch_context], axis=0)

    # Authentic neural network inference using the trained deepfake detection model
    raw_preds = loaded_model.model.predict(batch_dual, verbose=0)
    p_tight, p_context = raw_preds[0], raw_preds[1]

    # Evaluate prediction across scales:
    # If the broader context crop detects boundary/seam manipulation artifacts, emphasize it
    if float(p_context[1]) > float(p_tight[1]):
        raw_prediction = p_context
        active_batch = batch_context
        active_meta = meta_context
    else:
        raw_prediction = p_tight
        active_batch = batch_tight
        active_meta = meta_tight

    # Generate Grad-CAM Attention Heatmap
    heatmap_b64: str | None = None
    try:
        heatmap = generate_gradcam_heatmap(loaded_model.model, active_batch, pred_index=1)
        crop_coords = active_meta.get("crop_coords")
        overlay_rgb = overlay_heatmap(heatmap, original_bgr, crop_coords=crop_coords)
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
            "input_shape": list(active_batch.shape[1:]),
            "preprocessing": loaded_model.preprocessing,
            "legacy_binary_head": loaded_model.legacy_binary_head,
            "face_detected": active_meta.get("face_detected", False),
            "face_bbox": active_meta.get("bbox"),
            "heatmap_b64": heatmap_b64,
            "multi_scale_inspection": {
                "tight_fake_probability": float(p_tight[1]),
                "context_fake_probability": float(p_context[1]),
            },
        },
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if img_path.exists():
            print(f"Analyzing {img_path}...")
            result = predict_image_bytes(img_path.read_bytes(), filename=img_path.name)
            print(f"\nResult: {result.get('label')}")
            print(f"Confidence: {result.get('confidence'):.2%}")
            print(f"Details: {result}")
        else:
            print(f"File not found: {img_path}")
    else:
        print("=" * 60)
        print("Multimodal Deepfake Detection - Image Model")
        print("=" * 60)
        print("Usage:")
        print("  1. Analyze an image via CLI:")
        print("     python image_model/image_predict.py <path_to_image>")
        print("\n  2. Launch the Streamlit Web Application (Recommended):")
        print("     streamlit run streamlit_app.py")
        print("\n  3. Run the FastAPI REST Server:")
        print("     uvicorn main:app --reload")
        print("=" * 60)

