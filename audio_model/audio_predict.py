from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AUDIO_CONFIG, build_prediction_result

try:
    from .audio_model import load_audio_model
    from .audio_utils import extract_legacy_log_mel_batch, extract_mfcc_batch
except (ImportError, ValueError):
    from audio_model.audio_model import load_audio_model
    from audio_model.audio_utils import extract_legacy_log_mel_batch, extract_mfcc_batch



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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
        if audio_path.exists():
            print(f"Analyzing {audio_path}...")
            result = predict_audio_bytes(audio_path.read_bytes(), filename=audio_path.name)
            print(f"\nResult: {result.get('label')}")
            print(f"Confidence: {result.get('confidence'):.2%}")
            print(f"Details: {result}")
        else:
            print(f"File not found: {audio_path}")
    else:
        print("=" * 60)
        print("Multimodal Deepfake Detection - Audio Model")
        print("=" * 60)
        print("Usage:")
        print("  1. Analyze an audio file via CLI:")
        print("     python audio_model/audio_predict.py <path_to_audio>")
        print("\n  2. Launch the Streamlit Web Application (Recommended):")
        print("     streamlit run streamlit_app.py")
        print("\n  3. Run the FastAPI REST Server:")
        print("     uvicorn main:app --reload")
        print("=" * 60)


