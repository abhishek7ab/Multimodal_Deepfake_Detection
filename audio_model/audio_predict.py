from __future__ import annotations

from config import AUDIO_CONFIG, build_prediction_result

from .audio_model import load_audio_model
from .audio_utils import extract_legacy_log_mel_batch, extract_mfcc_batch


def predict_audio_bytes(audio_bytes: bytes, filename: str | None = None) -> dict:
    loaded_model = load_audio_model()
    if loaded_model.preprocessing == "mfcc_40":
        batch, meta = extract_mfcc_batch(audio_bytes, filename, return_meta=True)
        feature_type = "mfcc"
        num_features = AUDIO_CONFIG.num_mfcc
    else:
        batch, meta = extract_legacy_log_mel_batch(audio_bytes, filename, return_meta=True)
        feature_type = "log_mel_spectrogram"
        num_features = 128

    raw_prediction = loaded_model.model.predict(batch, verbose=0)[0]

    # DEBUG: Print raw model output
    print(f"[DEBUG AUDIO] Raw model output: {raw_prediction}")
    print(f"[DEBUG AUDIO] Model preprocessing: {loaded_model.preprocessing}")
    print(f"[DEBUG AUDIO] Model legacy_binary_head: {loaded_model.legacy_binary_head}")

    extra = {
        "filename": filename,
        "input_shape": list(batch.shape[1:]),
        "feature_type": feature_type,
        "num_features": num_features,
        "preprocessing": loaded_model.preprocessing,
        "legacy_binary_head": loaded_model.legacy_binary_head,
    }
    if meta:
        extra.update(meta)

    return build_prediction_result(
        modality="audio",
        raw_prediction=raw_prediction,
        threshold=AUDIO_CONFIG.threshold,
        model_source=loaded_model.source_path,
        extra=extra,
    )

