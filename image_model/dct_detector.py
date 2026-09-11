"""
dct_detector.py
===============
Frequency-domain artifact detector based on Discrete Cosine Transform (DCT) analysis.

WHY THIS EXISTS
---------------
The primary CNN (deepfake_cnn.keras) was trained on the 140 k real-vs-face-swap
Kaggle dataset.  It excels at detecting face-swap / DeepFaceLab forgeries but
is *not* effective against modern diffusion-model images (Stable Diffusion,
Midjourney, DALL-E 3, etc.), which look statistically very natural to a face-swap
detector.

Diffusion models leave systematic spectral fingerprints in the DCT domain that
are largely invisible in the pixel domain.  This module analyses those fingerprints
and produces a supplementary "AI-synthesis probability" score.

APPROACH
--------
1. Convert the image to grayscale YCbCr luma channel.
2. Compute the 2-D DCT of the full image.
3. Shift the DC component to the centre (log-magnitude spectrum).
4. Measure several spectral statistics known to differ between real and
   AI-synthesised images:
   - High-frequency energy ratio (real photos have a steep 1/f roll-off;
     diffusion images have a flatter high-freq tail due to denoising artefacts).
   - Azimuthal symmetry deviation (diffusion images are often more isotropic).
   - Peak-to-average ratio in the mid-frequency band (GAN grid artefacts).
   - Blocking-artefact energy at 8-px / 16-px frequencies (absent in diffusion).
5. Combine these into a heuristic 0-1 score, then calibrate with a sigmoid.

NOTE: This is a *heuristic* detector.  It is NOT a trained classifier and will
have false-positive / false-negative rates.  It should be treated as supplementary
evidence alongside the CNN verdict.
"""
from __future__ import annotations

import warnings
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Return float32 grayscale image in [0, 1]."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


def _compute_dct_spectrum(gray: np.ndarray) -> np.ndarray:
    """
    Compute the log-magnitude of the 2-D DCT spectrum of a grayscale image.
    Returns a 2-D array the same size as *gray*.
    """
    # Resize to a fixed square to make statistics comparable across images
    h, w = gray.shape
    side = min(h, w, 512)
    side = side - (side % 8)  # ensure multiple of 8 for block analysis
    resized = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)

    dct = cv2.dct(resized)
    # Log magnitude (avoid log(0))
    log_mag = np.log1p(np.abs(dct))
    return log_mag


def _radial_profile(log_mag: np.ndarray) -> np.ndarray:
    """Return the mean log-magnitude as a function of radial frequency."""
    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.hypot(X - cx, Y - cy).astype(np.int32)
    r_max = min(cy, cx)
    profile = np.bincount(r.ravel(), weights=log_mag.ravel(), minlength=r_max + 1)
    counts = np.bincount(r.ravel(), minlength=r_max + 1).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    return (profile / counts)[:r_max]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _hf_energy_ratio(log_mag: np.ndarray) -> float:
    """
    Ratio of high-freq energy to total energy in the DCT.
    Real images → steep 1/f drop → low HF ratio.
    Diffusion images → flatter tail → higher HF ratio.
    """
    h, w = log_mag.shape
    # High-frequency corner: top-right 25 % × 25 % of the DCT matrix
    hf_h, hf_w = h * 3 // 4, w * 3 // 4
    hf_energy = float(log_mag[hf_h:, hf_w:].mean())
    total_energy = float(log_mag.mean())
    if total_energy <= 0:
        return 0.0
    return hf_energy / total_energy


def _spectral_flatness(profile: np.ndarray) -> float:
    """
    Wiener spectral flatness of the radial profile.
    White noise → 1.0.  Natural images (1/f) → near 0.
    Diffusion tends to be flatter than real photos.
    """
    eps = 1e-8
    geo_mean = float(np.exp(np.mean(np.log(np.abs(profile) + eps))))
    arith_mean = float(np.mean(np.abs(profile)) + eps)
    return geo_mean / arith_mean


def _mid_freq_peak_ratio(log_mag: np.ndarray) -> float:
    """
    Check for anomalous peaks in the mid-frequency band [10 %, 40 %] of Nyquist.
    GAN-generated images often show grid artefacts here.
    """
    h, w = log_mag.shape
    r0, r1 = int(min(h, w) * 0.10), int(min(h, w) * 0.40)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.hypot(X - cx, Y - cy)
    mask = (r >= r0) & (r <= r1)
    band = log_mag[mask]
    if band.size == 0:
        return 0.0
    return float(band.max() / (band.mean() + 1e-8))


def _blocking_artefact_score(gray: np.ndarray, block_size: int = 8) -> float:
    """
    JPEG-style blocking is absent in diffusion images (they don't go through JPEG
    compression in the synthesis pipeline unless saved as JPEG).
    High score → JPEG-like blocking present → likely real photo or face-swap.
    Low score → no blocking → might be diffusion.
    """
    h, w = gray.shape
    h = h - (h % block_size)
    w = w - (w % block_size)
    crop = gray[:h, :w]

    # Horizontal block boundaries
    horiz = np.abs(crop[block_size - 1:h - 1:block_size, :] -
                   crop[block_size::block_size, :]).mean()
    # Vertical block boundaries
    vert = np.abs(crop[:, block_size - 1:w - 1:block_size] -
                  crop[:, block_size::block_size]).mean()

    # Interior gradients for normalisation
    interior_h = np.abs(np.diff(crop, axis=0)).mean()
    interior_v = np.abs(np.diff(crop, axis=1)).mean()
    norm = (interior_h + interior_v) / 2.0 + 1e-8

    return float((horiz + vert) / 2.0 / norm)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def _sigmoid(x: float, scale: float = 1.0, bias: float = 0.0) -> float:
    """Numerically stable sigmoid."""
    z = float(np.clip(-scale * (x - bias), -500, 500))
    return 1.0 / (1.0 + np.exp(z))


def analyse_dct_artifacts(image_bgr: np.ndarray) -> dict[str, Any]:
    """
    Run the DCT-based AI-synthesis detector on a BGR image array.

    Returns
    -------
    dict with keys:
        ai_synthesis_probability  : float in [0, 1]
        hf_energy_ratio           : float (feature)
        spectral_flatness         : float (feature)
        mid_freq_peak_ratio       : float (feature)
        blocking_score            : float (feature)
        note                      : str   (human-readable summary)
    """
    try:
        gray = _to_gray(image_bgr)
        h, w = gray.shape
        if h < 32 or w < 32:
            return _error_result("Image too small for DCT analysis.")

        log_mag = _compute_dct_spectrum(gray)
        profile = _radial_profile(log_mag)

        hf_ratio = _hf_energy_ratio(log_mag)
        flatness = _spectral_flatness(profile)
        mid_peak = _mid_freq_peak_ratio(log_mag)
        blocking = _blocking_artefact_score(gray)

        # ------------------------------------------------------------------
        # Heuristic scoring
        # ------------------------------------------------------------------
        # Each feature contributes a "suspicion" value in [0, 1].
        # Higher suspicion → more likely AI-generated.
        #
        # Calibration was hand-tuned on a small reference set; it will not be
        # perfectly calibrated but gives a meaningful relative signal.

        # HF energy: real ≈ 0.55-0.70; diffusion ≈ 0.72-0.85
        hf_susp = _sigmoid(hf_ratio, scale=35.0, bias=0.73)

        # Spectral flatness: real ≈ 0.75-0.88; diffusion ≈ 0.90-0.98
        flat_susp = _sigmoid(flatness, scale=30.0, bias=0.91)

        # Mid-freq peak ratio: GAN grids → high peaks; real / diffusion lower
        # Inverted: high peaks → MORE likely real/GAN face-swap, not diffusion
        peak_susp = _sigmoid(mid_peak, scale=-0.6, bias=6.0)

        # Blocking: real JPEG photos → higher blocking → less AI-like
        block_susp = _sigmoid(blocking, scale=-15.0, bias=0.08)

        # Weighted combination
        score = 0.35 * hf_susp + 0.35 * flat_susp + 0.20 * peak_susp + 0.10 * block_susp

        # Build human-readable note
        if score >= 0.70:
            note = (
                "High spectral anomaly score — consistent with AI-generative synthesis "
                "(diffusion model / GAN). CNN verdict may be unreliable."
            )
        elif score >= 0.45:
            note = (
                "Moderate spectral anomaly — image may have been post-processed or "
                "generated by AI. Treat CNN verdict with caution."
            )
        else:
            note = "Low spectral anomaly — no strong frequency-domain synthesis markers detected."

        return {
            "ai_synthesis_probability": round(float(score), 4),
            "hf_energy_ratio": round(hf_ratio, 4),
            "spectral_flatness": round(flatness, 4),
            "mid_freq_peak_ratio": round(mid_peak, 4),
            "blocking_score": round(blocking, 4),
            "note": note,
        }

    except Exception as exc:  # pylint: disable=broad-except
        warnings.warn(f"[DCT detector] Analysis failed: {exc}", stacklevel=2)
        return _error_result(str(exc))


def _error_result(reason: str) -> dict[str, Any]:
    return {
        "ai_synthesis_probability": None,
        "hf_energy_ratio": None,
        "spectral_flatness": None,
        "mid_freq_peak_ratio": None,
        "blocking_score": None,
        "note": f"DCT analysis unavailable: {reason}",
    }
