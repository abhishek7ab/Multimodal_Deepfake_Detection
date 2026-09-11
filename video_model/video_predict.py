from __future__ import annotations

import numpy as np

from config import VIDEO_CONFIG, build_prediction_result
from image_model.image_model import load_image_model
from image_model.image_utils import preprocess_image_array

from .video_utils import extract_audio_track_from_video, extract_video_frames


def predict_video_bytes(video_bytes: bytes, filename: str | None = None) -> dict:
    loaded_image_model = load_image_model()
    frames = extract_video_frames(video_bytes, filename)

    frame_batches = []
    face_detections_count = 0
    for frame_bgr in frames:
        frame_tensor, meta = preprocess_image_array(
            frame_bgr,
            mode=loaded_image_model.preprocessing,
            crop_face=True,
        )
        frame_batches.append(frame_tensor[0])
        if meta.get("face_detected", False):
            face_detections_count += 1

    batch = np.stack(frame_batches, axis=0).astype(np.float32)
    frame_predictions = loaded_image_model.model.predict(batch, verbose=0)
    average_prediction = np.mean(frame_predictions, axis=0)

    # Extract audio track for multimodal fusion
    audio_bytes = extract_audio_track_from_video(video_bytes, filename)
    audio_result = None
    audio_detected = False
    if audio_bytes is not None and len(audio_bytes) > 2000:
        try:
            from audio_model.audio_predict import predict_audio_bytes

            audio_result = predict_audio_bytes(audio_bytes, filename=f"{filename or 'video'}.wav")
            audio_detected = True
        except Exception as exc:
            print(f"[WARN VIDEO] Could not analyze extracted audio track: {exc}")

    # Calculate visual fake probability
    if average_prediction.size == 2:
        vis_fake_prob = float(average_prediction[1])
    else:
        vis_fake_prob = float(average_prediction[0])

    # Multimodal Fusion: Combine visual (60%) and acoustic (40%) forensic signals
    if audio_result and "probability_fake" in audio_result:
        aud_fake_prob = float(audio_result["probability_fake"])
        fused_fake_prob = float(np.clip(0.60 * vis_fake_prob + 0.40 * aud_fake_prob, 0.0, 1.0))
        final_prediction = np.array([1.0 - fused_fake_prob, fused_fake_prob], dtype=np.float32)
    else:
        aud_fake_prob = None
        fused_fake_prob = vis_fake_prob
        final_prediction = average_prediction

    # Per-frame fake probability sequence for timeline charts
    if frame_predictions.shape[-1] == 2:
        per_frame_fake = [float(p[1]) for p in frame_predictions]
    else:
        per_frame_fake = [float(p[0]) for p in frame_predictions]

    return build_prediction_result(
        modality="video",
        raw_prediction=final_prediction,
        threshold=VIDEO_CONFIG.threshold,
        model_source=f"multimodal_fusion::visual({loaded_image_model.source_path})+audio",
        extra={
            "filename": filename,
            "frames_used": int(batch.shape[0]),
            "faces_detected_in_frames": face_detections_count,
            "frame_shape": list(batch.shape[1:]),
            "preprocessing": loaded_image_model.preprocessing,
            "frame_probabilities": np.asarray(frame_predictions, dtype=np.float32).tolist(),
            "per_frame_fake_probabilities": per_frame_fake,
            "audio_detected": audio_detected,
            "visual_probability_fake": vis_fake_prob,
            "audio_probability_fake": aud_fake_prob,
            "multimodal_fusion_applied": bool(audio_detected),
            "audio_details": audio_result,
        },
    )
