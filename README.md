# Multimodal Deepfake Detection System


[[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multimodaldeepfakedetection-fwyv3uecaawyswbwcais8c.streamlit.app/)](https://multimodaldeepfakedetection-fwyv3uecaawyswbwcais8c.streamlit.app/)

## 🌐 Live Deployment
🚀 **Live Streamlit App:** [[DeepGuard | Multimodal Deepfake Forensics](https://multimodaldeepfakedetection-fwyv3uecaawyswbwcais8c.streamlit.app/)](https://multimodaldeepfakedetection-fwyv3uecaawyswbwcais8c.streamlit.app/)

This project detects deepfakes across three modalities:

- Image
- Audio
- Video

The repository includes:

- A FastAPI app with unified prediction endpoints in `main.py`
- A Streamlit interface in `streamlit_app.py`
- Separate modality modules under `image_model/`, `audio_model/`, and `video_model/`
- Saved model files for legacy image and audio inference

## Project Structure

```text
.
|-- audio_model/
|-- frontend/
|-- image_model/
|-- video_model/
|-- config.py
|-- main.py
|-- requirements.txt
`-- streamlit_app.py
```

## How It Works

- Image detection loads a Keras model from `image_model/`
- Audio detection extracts MFCC or log-mel features, then runs a Keras model from `audio_model/`
- Video detection samples frames from the uploaded video and reuses the image model as a frame-level ensemble

The app normalizes predictions into a common two-class output:

- `REAL`
- `FAKE`

## Requirements

- Python 3.10 or newer recommended
- `pip`
- Model files already present in this repository

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

This is the easiest way to use the project locally.

```bash
streamlit run streamlit_app.py
```

The UI lets you upload:

- Images: `jpg`, `jpeg`, `png`, `bmp`
- Audio: `wav`, `mp3`, `m4a`, `flac`
- Video: `mp4`, `avi`, `mov`, `mkv`

## Run the FastAPI App

Start the unified API:

```bash
uvicorn main:app --reload
```

Useful routes:

- `GET /health`
- `POST /image/predict`
- `POST /audio/predict`
- `POST /video/predict`

Swagger docs will be available at:

```text
http://127.0.0.1:8000/docs
```

## Example API Usage

Image prediction:

```bash
curl -X POST "http://127.0.0.1:8000/image/predict" -F "file=@sample.jpg"
```

Audio prediction:

```bash
curl -X POST "http://127.0.0.1:8000/audio/predict" -F "file=@sample.wav"
```

Video prediction:

```bash
curl -X POST "http://127.0.0.1:8000/video/predict" -F "file=@sample.mp4"
```

Typical response fields:

```json
{
  "modality": "image",
  "label": "REAL",
  "class_index": 0,
  "confidence": 0.91,
  "threshold": 0.5,
  "raw_probabilities": [0.91, 0.09],
  "model_source": "image_model/deepfake_cnn.keras"
}
```

## Configuration

Runtime settings are defined in `config.py` and can be overridden with environment variables.

Important options include:

- `IMAGE_WIDTH`
- `IMAGE_HEIGHT`
- `IMAGE_THRESHOLD`
- `AUDIO_SAMPLE_RATE`
- `AUDIO_TARGET_SAMPLES`
- `AUDIO_NUM_MFCC`
- `AUDIO_MAX_FRAMES`
- `AUDIO_THRESHOLD`
- `VIDEO_FRAME_COUNT`
- `VIDEO_FRAME_STRIDE`
- `VIDEO_THRESHOLD`

## Model Notes

The code supports both newer and legacy model naming conventions.

- Image loader checks for `image_model/efficientnet_deepfake.keras` first, then falls back to `image_model/deepfake_cnn.keras`
- Audio loader checks for `audio_model/audio_classifier.keras`, then falls back to legacy weights or legacy model files
- Video inference currently reuses the image model on sampled video frames

In the current folder contents, the legacy image and audio model files are present.

## Additional Files

- `frontend/app.py` is a separate Streamlit frontend that expects individual image/audio/video services running on container-style hostnames
- `image_model/app.py`, `audio_model/app.py`, and `video_model/app.py` expose per-modality FastAPI services with `/predict` endpoints

## Notes

- Video inference may be slower because frames are sampled and evaluated one by one
- Audio and video uploads are temporarily written to disk during preprocessing
- Some notebook files are included for experimentation and training history
