from __future__ import annotations

import base64
import os
import subprocess
import tempfile

import cv2
import numpy as np
import soundfile as sf
import tensorflow as tf

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

from config import AUDIO_CONFIG


def _temp_suffix(filename: str | None) -> str:
    if not filename:
        return ".wav"
    _, extension = os.path.splitext(filename)
    return extension or ".wav"


def _load_audio_via_ffmpeg(file_path: str) -> tuple[np.ndarray, int]:
    """Fallback decoder for non-WAV/FLAC formats (e.g. M4A, AAC, WebM) using bundled FFmpeg."""
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_wav = file_path + "_converted.wav"
    try:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", file_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(AUDIO_CONFIG.sample_rate),
            "-ac", "1",
            out_wav,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if result.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 100:
            waveform, sample_rate = sf.read(out_wav, dtype="float32")
            return waveform, sample_rate
        raise RuntimeError(
            f"FFmpeg decoding failed: {result.stderr.decode('utf-8', errors='ignore')}"
        )
    finally:
        if os.path.exists(out_wav):
            try:
                os.unlink(out_wav)
            except OSError:
                pass


def load_audio_from_bytes(
    audio_bytes: bytes,
    filename: str | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    suffix = _temp_suffix(filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:
        try:
            waveform, sample_rate = sf.read(temp_path, dtype="float32")
        except Exception:
            waveform, sample_rate = _load_audio_via_ffmpeg(temp_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    if waveform.size == 0:
        raise ValueError("Could not decode the uploaded audio.")

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    orig_sample_rate = int(sample_rate)
    orig_duration = float(len(waveform) / sample_rate) if sample_rate > 0 else 0.0

    if sample_rate != AUDIO_CONFIG.sample_rate:
        waveform = _resample_waveform(waveform, sample_rate, AUDIO_CONFIG.sample_rate)
        sample_rate = AUDIO_CONFIG.sample_rate

    waveform = waveform.astype(np.float32)
    peak = float(np.max(np.abs(waveform)))
    if peak > 1.0:
        waveform = waveform / peak

    if return_meta:
        meta = {
            "original_sample_rate": orig_sample_rate,
            "target_sample_rate": int(sample_rate),
            "duration_seconds": round(orig_duration, 2),
            "peak_amplitude": round(peak, 4),
            "rms_energy": round(float(np.sqrt(np.mean(waveform ** 2))), 5),
        }
        return waveform, meta

    return waveform


def pad_or_trim_audio(waveform: np.ndarray) -> np.ndarray:
    if waveform.shape[0] < AUDIO_CONFIG.target_samples:
        pad_width = AUDIO_CONFIG.target_samples - waveform.shape[0]
        waveform = np.pad(waveform, (0, pad_width), mode="constant")
    else:
        waveform = waveform[: AUDIO_CONFIG.target_samples]
    return waveform.astype(np.float32)


def _resample_waveform(waveform: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    if original_rate == target_rate:
        return waveform.astype(np.float32)

    target_length = max(1, int(round(len(waveform) * float(target_rate) / float(original_rate))))
    original_positions = np.linspace(0.0, 1.0, num=len(waveform), endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_positions, original_positions, waveform).astype(np.float32)


def generate_spectrogram_base64(feature_matrix: np.ndarray) -> str | None:
    """Renders a 2D spectral matrix as a colormapped PNG image encoded in base64."""
    try:
        mat = np.squeeze(feature_matrix)
        if mat.ndim != 2:
            return None
        min_val, max_val = float(mat.min()), float(mat.max())
        if max_val - min_val > 1e-6:
            norm = (mat - min_val) / (max_val - min_val)
        else:
            norm = np.zeros_like(mat)
        # Flip so low frequencies appear at the bottom
        img_uint8 = np.flipud((norm * 255.0).astype(np.uint8))
        color_img = cv2.applyColorMap(img_uint8, cv2.COLORMAP_MAGMA)
        resized = cv2.resize(color_img, (540, 160), interpolation=cv2.INTER_CUBIC)
        success, buf = cv2.imencode(".png", resized)
        if success:
            return base64.b64encode(buf).decode("ascii")
        return None
    except Exception as exc:
        print(f"[WARN] Failed to generate spectrogram image: {exc}")
        return None


def extract_mfcc_batch(
    audio_bytes: bytes,
    filename: str | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    audio_res = load_audio_from_bytes(audio_bytes, filename, return_meta=True)
    waveform, meta = audio_res  # type: ignore[misc]
    waveform_padded = pad_or_trim_audio(waveform)

    if _HAS_LIBROSA:
        mfcc = librosa.feature.mfcc(
            y=waveform_padded,
            sr=AUDIO_CONFIG.sample_rate,
            n_mfcc=AUDIO_CONFIG.num_mfcc,
        ).astype(np.float32)
        mean = float(np.mean(mfcc))
        std = float(np.std(mfcc))
        mfcc = (mfcc - mean) / (std + 1e-6)

        if mfcc.shape[1] < AUDIO_CONFIG.max_frames:
            pad_width = AUDIO_CONFIG.max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, : AUDIO_CONFIG.max_frames]
    else:
        waveform_tensor = tf.convert_to_tensor(waveform_padded, dtype=tf.float32)
        stft = tf.signal.stft(
            waveform_tensor,
            frame_length=400,
            frame_step=160,
            fft_length=512,
            window_fn=tf.signal.hann_window,
        )
        spectrogram = tf.abs(stft) ** 2
        mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=64,
            num_spectrogram_bins=int(spectrogram.shape[-1]),
            sample_rate=AUDIO_CONFIG.sample_rate,
            lower_edge_hertz=20.0,
            upper_edge_hertz=AUDIO_CONFIG.sample_rate / 2.0,
        )
        mel_spectrogram = tf.matmul(spectrogram, mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., : AUDIO_CONFIG.num_mfcc]
        mfcc = tf.transpose(mfcc).numpy().astype(np.float32)

        mean = float(np.mean(mfcc))
        std = float(np.std(mfcc))
        mfcc = (mfcc - mean) / (std + 1e-6)

        if mfcc.shape[1] < AUDIO_CONFIG.max_frames:
            pad_width = AUDIO_CONFIG.max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, : AUDIO_CONFIG.max_frames]

    batch = np.expand_dims(mfcc[..., np.newaxis], axis=0).astype(np.float32)
    meta["spectrogram_b64"] = generate_spectrogram_base64(mfcc)

    if return_meta:
        return batch, meta
    return batch


def extract_legacy_log_mel_batch(
    audio_bytes: bytes,
    filename: str | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    audio_res = load_audio_from_bytes(audio_bytes, filename, return_meta=True)
    waveform, meta = audio_res  # type: ignore[misc]
    waveform_padded = pad_or_trim_audio(waveform)

    target_frames = 63
    if _HAS_LIBROSA:
        # Matches training pipeline from audio_deepfake.ipynb exactly:
        # librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128) + power_to_db
        spec = librosa.feature.melspectrogram(
            y=waveform_padded,
            sr=AUDIO_CONFIG.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )
        log_mel = librosa.power_to_db(spec).astype(np.float32)
        if log_mel.shape[1] < target_frames:
            pad_width = target_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
        else:
            log_mel = log_mel[:, :target_frames]
    else:
        # Calibrated TensorFlow fallback if librosa is unavailable
        waveform_tensor = tf.convert_to_tensor(waveform_padded, dtype=tf.float32)
        stft = tf.signal.stft(
            waveform_tensor,
            frame_length=2048,
            frame_step=512,
            fft_length=2048,
            window_fn=tf.signal.hann_window,
        )
        spectrogram = tf.abs(stft) ** 2
        mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=128,
            num_spectrogram_bins=int(spectrogram.shape[-1]),
            sample_rate=AUDIO_CONFIG.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=AUDIO_CONFIG.sample_rate / 2.0,
        )
        mel_spectrogram = tf.matmul(spectrogram, mel_weight_matrix)
        # Convert power to decibels: 10 * log10(max(S, 1e-10))
        log_mel_spectrogram = 10.0 * tf.math.log(tf.maximum(mel_spectrogram, 1e-10)) / tf.math.log(10.0)
        log_mel = tf.transpose(log_mel_spectrogram).numpy().astype(np.float32)

        if log_mel.shape[1] < target_frames:
            pad_width = target_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
        else:
            log_mel = log_mel[:, :target_frames]

    batch = np.expand_dims(log_mel[..., np.newaxis], axis=0).astype(np.float32)
    meta["spectrogram_b64"] = generate_spectrogram_base64(log_mel)

    if return_meta:
        return batch, meta
    return batch

