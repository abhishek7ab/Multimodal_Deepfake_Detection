from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from audio_model.audio_predict import predict_audio_bytes
from image_model.image_predict import predict_image_bytes
from video_model.video_predict import predict_video_bytes


app = FastAPI(title="Multimodal Deepfake Detection API", version="2.0.0")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


async def _read_upload(file: UploadFile) -> bytes:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return payload


@app.post("/image/predict")
async def image_predict(file: UploadFile = File(...)) -> dict:
    try:
        return predict_image_bytes(await _read_upload(file), filename=file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/audio/predict")
async def audio_predict(file: UploadFile = File(...)) -> dict:
    try:
        return predict_audio_bytes(await _read_upload(file), filename=file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/video/predict")
async def video_predict(file: UploadFile = File(...)) -> dict:
    try:
        return predict_video_bytes(await _read_upload(file), filename=file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
