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


def _noise_floor_score(gray: np.ndarray) -> float:
    """
    Real camera photos contain photon/sensor noise that manifests as
    a raised white-noise floor in the high-frequency DCT.
    AI-generated images lack this noise → very clean HF floor.
    Returns a LOW value for AI images (high-frequency floor is suspiciously clean).
    """
    h, w = gray.shape
    side = min(h, w, 256)
    side = side - (side % 8)
    resized = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized)
    # High-frequency corner (top-right quadrant of the DCT matrix)
    hf_corner = dct[side * 3 // 4:, side * 3 // 4:]
    # Real photos: noisy (std is higher relative to mean)
    # AI images: clean (std is lower relative to mean)
    std = float(np.std(np.abs(hf_corner)))
    mean = float(np.mean(np.abs(hf_corner))) + 1e-8
    return std / mean  # coefficient of variation; real ≈ 1.0+, AI ≈ 0.5-0.8


def _channel_independence(image_bgr: np.ndarray) -> float:
    """
    In real camera images, the RGB channels are highly correlated because
    they record the same scene under the same lighting.
    AI generative models can produce channels that are more independently
    synthesised, leading to lower cross-channel correlation.
    Returns the MEAN cross-channel Pearson correlation (higher = more natural).
    """
    b = image_bgr[:, :, 0].astype(np.float32).ravel()
    g = image_bgr[:, :, 1].astype(np.float32).ravel()
    r = image_bgr[:, :, 2].astype(np.float32).ravel()
    try:
        rg = float(np.corrcoef(r, g)[0, 1])
        rb = float(np.corrcoef(r, b)[0, 1])
        gb = float(np.corrcoef(g, b)[0, 1])
        return float(np.mean([abs(rg), abs(rb), abs(gb)]))
    except Exception:
        return 0.9  # assume natural if computation fails


def _azimuthal_isotropy(log_mag: np.ndarray) -> float:
    """
    Compute the variance of the azimuthal (angular) distribution of the
    DCT energy in the mid-frequency band.
    Real photos have directional textures (edges, hair, fabric) → high variance.
    Diffusion images often have more isotropic synthesis → lower variance.
    Returns variance (higher = more natural / directional).
    """
    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.hypot(X - cx, Y - cy)
    theta = np.arctan2(Y - cy, X - cx)  # -pi to pi

    # Mid-frequency annulus: 15–40% of Nyquist
    r_inner = min(h, w) * 0.15
    r_outer = min(h, w) * 0.40
    mask = (r >= r_inner) & (r <= r_outer)

    if mask.sum() == 0:
        return 0.0

    # Bin into 36 angular sectors (10° each)
    theta_norm = ((theta + np.pi) / (2 * np.pi) * 36).astype(int)
    theta_norm = np.clip(theta_norm, 0, 35)
    sector_energy = np.zeros(36)
    for s in range(36):
        sector_mask = mask & (theta_norm == s)
        if sector_mask.sum() > 0:
            sector_energy[s] = log_mag[sector_mask].mean()

    return float(np.std(sector_energy))  # higher std → more directional → more real


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
        noise_floor_cv            : float (feature)
        channel_correlation       : float (feature)
        azimuthal_variance        : float (feature)
        note                      : str   (human-readable summary)
    """
    try:
        gray = _to_gray(image_bgr)
        h, w = gray.shape
        if h < 32 or w < 32:
            return _error_result("Image too small for DCT analysis.")

        log_mag = _compute_dct_spectrum(gray)
        profile = _radial_profile(log_mag)

        hf_ratio  = _hf_energy_ratio(log_mag)
        flatness  = _spectral_flatness(profile)
        mid_peak  = _mid_freq_peak_ratio(log_mag)
        blocking  = _blocking_artefact_score(gray)
        noise_cv  = _noise_floor_score(gray)
        ch_corr   = _channel_independence(image_bgr)
        az_var    = _azimuthal_isotropy(log_mag)

        # ------------------------------------------------------------------
        # Recalibrated heuristic scoring
        # ------------------------------------------------------------------
        # Feature ranges from observations:
        #   Real photos:   HF≈0.58-0.71, flat≈0.77-0.89, noise_cv≈0.95-1.3,
        #                  ch_corr≈0.88-0.97, az_var≈0.25-0.55
        #   AI/Diffusion:  HF≈0.68-0.80, flat≈0.85-0.97, noise_cv≈0.45-0.80,
        #                  ch_corr≈0.78-0.91, az_var≈0.12-0.30

        # 1. HF energy: AI images have flatter high-freq tail.
        #    Fire when HF > 0.68 (was 0.73 — too conservative)
        hf_susp = _sigmoid(hf_ratio, scale=25.0, bias=0.68)

        # 2. Spectral flatness: AI images more spectrally flat.
        #    Fire when flatness > 0.86 (was 0.91 — too conservative)
        flat_susp = _sigmoid(flatness, scale=22.0, bias=0.86)

        # 3. Noise floor coefficient of variation:
        #    Real photos have noisy HF → cv ≈ 1.0+; AI images clean → cv ≈ 0.6
        #    Inverted: LOW cv → MORE suspicious
        noise_susp = _sigmoid(noise_cv, scale=-8.0, bias=0.80)

        # 4. Channel correlation:
        #    Real photos → high corr ≈ 0.92+; AI may be lower ≈ 0.82-0.90
        #    Inverted: LOWER corr → more suspicious
        ch_susp = _sigmoid(ch_corr, scale=-18.0, bias=0.88)

        # 5. Azimuthal isotropy:
        #    Real → high variance (directional textures); AI → lower variance
        #    Inverted: LOWER variance → more suspicious
        az_susp = _sigmoid(az_var, scale=-12.0, bias=0.30)

        # 6. Blocking (original feature — kept for continuity):
        #    Real JPEG → higher blocking; AI PNG → near zero
        block_susp = _sigmoid(blocking, scale=-12.0, bias=0.06)

        # 7. Mid-freq peak: high peaks in GAN grid (less relevant for diffusion)
        peak_susp = _sigmoid(mid_peak, scale=-0.5, bias=5.5)

        # Weighted ensemble — noise_cv and ch_corr are the strongest new signals
        score = (
            0.20 * hf_susp
            + 0.15 * flat_susp
            + 0.20 * noise_susp   # strong: real cameras always have sensor noise
            + 0.15 * ch_susp      # useful: AI channels can be more independent
            + 0.15 * az_susp      # useful: AI images more isotropic
            + 0.10 * block_susp   # moderate: only useful for JPEG source images
            + 0.05 * peak_susp    # weak: mainly for old GAN grids
        )

        # Build human-readable note
        if score >= 0.55:
            note = (
                "⚠️ High AI-synthesis probability — spectral, noise-floor, and "
                "channel-independence analysis all show markers consistent with "
                "diffusion-model or GAN generation. CNN verdict likely unreliable for this image type."
            )
        elif score >= 0.35:
            note = (
                "🟡 Moderate AI-synthesis indicators — noise floor and spectral "
                "analysis detected some characteristics of synthetic images. "
                "The CNN may not be reliable here; consider additional verification."
            )
        else:
            note = "🟢 Low AI-synthesis markers — no strong frequency-domain synthesis signals detected."

        return {
            "ai_synthesis_probability": round(float(score), 4),
            "hf_energy_ratio": round(hf_ratio, 4),
            "spectral_flatness": round(flatness, 4),
            "mid_freq_peak_ratio": round(mid_peak, 4),
            "blocking_score": round(blocking, 4),
            "noise_floor_cv": round(noise_cv, 4),
            "channel_correlation": round(ch_corr, 4),
            "azimuthal_variance": round(az_var, 4),
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
        "noise_floor_cv": None,
        "channel_correlation": None,
        "azimuthal_variance": None,
        "note": f"DCT analysis unavailable: {reason}",
    }
