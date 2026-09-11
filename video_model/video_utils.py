from __future__ import annotations

import os
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np

from config import VIDEO_CONFIG


def _temp_suffix(filename: str | None) -> str:
    if not filename:
        return ".mp4"
    _, extension = os.path.splitext(filename)
    return extension or ".mp4"


def extract_audio_track_from_video(video_bytes: bytes, filename: str | None = None) -> bytes | None:
    """
    Extracts the audio track from video bytes as 16kHz mono WAV using bundled FFmpeg.
    Returns None if no audio track exists or extraction fails.
    """
    suffix = _temp_suffix(filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as in_f:
        in_f.write(video_bytes)
        in_path = in_f.name

    out_path = in_path + ".wav"
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", in_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            out_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            with open(out_path, "rb") as f:
                return f.read()
        return None
    except Exception as exc:
        print(f"[WARN] Failed to extract audio track from video: {exc}")
        return None
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def extract_video_frames(video_bytes: bytes, filename: str | None = None) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=_temp_suffix(filename)) as temp_file:
        temp_file.write(video_bytes)
        temp_path = temp_file.name

    capture = cv2.VideoCapture(temp_path)
    frames: list[np.ndarray] = []

    try:
        while len(frames) < VIDEO_CONFIG.frame_count:
            grabbed = capture.grab()
            if not grabbed:
                break

            current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if current_frame % VIDEO_CONFIG.frame_stride != 0:
                continue

            success, frame_bgr = capture.retrieve()
            if not success or frame_bgr is None:
                continue
            frames.append(frame_bgr)
    finally:
        capture.release()
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    if not frames:
        raise ValueError("Could not decode any frames from the uploaded video.")

    if len(frames) < VIDEO_CONFIG.frame_count:
        frames.extend([frames[-1].copy() for _ in range(VIDEO_CONFIG.frame_count - len(frames))])

    return np.asarray(frames, dtype=np.uint8)
