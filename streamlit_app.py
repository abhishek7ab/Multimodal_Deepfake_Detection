import base64
import hashlib
import json
import time
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="DeepGuard | Multimodal Deepfake Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* Global Reset & Dark Canvas */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #111a2e 0%, #090d16 65%, #05070c 100%);
    background-attachment: fixed;
}

header[data-testid="stHeader"] {
    background: rgba(9, 13, 22, 0.75) !important;
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.block-container {
    max-width: 1240px;
    padding-top: 5rem !important;
    padding-bottom: 4rem !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: #090d16 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Outfit', sans-serif;
    color: #f8fafc;
}

/* Hero Section */
.hero {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(30, 41, 59, 0.75) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 24px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 0 0 30px -5px rgba(56, 189, 248, 0.12);
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -10%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.badge-row {
    display: flex;
    gap: 0.6rem;
    align-items: center;
    margin-bottom: 0.8rem;
}

.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.32rem 0.85rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38bdf8;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    display: inline-block;
}

.hero h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0.4rem 0 0.7rem;
    background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 40%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    line-height: 1.65;
    max-width: 820px;
    margin: 0;
}

/* Feature Grid */
.grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.2rem 0 1.8rem;
}

.feature-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
    transition: all 0.25s ease;
}

.feature-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
}

.feature-card .f-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    display: inline-block;
}

.feature-card b {
    display: block;
    font-family: 'Outfit', sans-serif;
    color: #f8fafc;
    font-size: 1.05rem;
    margin-bottom: 0.25rem;
}

.feature-card span {
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.5;
    display: block;
}

/* Glass Panels */
.glass-panel {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 22px;
    padding: 1.6rem;
    box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.5);
    margin-bottom: 1.2rem;
    backdrop-filter: blur(16px);
}

.panel-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.panel-sub {
    color: #94a3b8;
    font-size: 0.88rem;
    margin: 0.25rem 0 1.2rem;
}

/* File Uploader Customization */
[data-testid="stFileUploader"] {
    background: rgba(10, 14, 23, 0.65) !important;
    border: 1.5px dashed rgba(56, 189, 248, 0.35) !important;
    border-radius: 18px !important;
    padding: 1.2rem !important;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.04) !important;
}

[data-testid="stFileUploader"] section {
    border: none !important;
    background: transparent !important;
}

/* Primary Action Button */
.stButton>button {
    width: 100%;
    min-height: 52px;
    border: none;
    border-radius: 14px;
    background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
    color: #ffffff;
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    box-shadow: 0 8px 25px -5px rgba(79, 70, 229, 0.5);
    transition: all 0.25s ease;
    margin-top: 0.6rem;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #0369a1 0%, #4338ca 100%);
    box-shadow: 0 10px 30px -4px rgba(79, 70, 229, 0.65);
    transform: translateY(-1.5px);
    color: #ffffff !important;
}

/* Result Cards */
.result-card {
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    border: 1px solid;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}

.result-card.real {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(2, 44, 34, 0.75) 100%);
    border-color: rgba(16, 185, 129, 0.4);
    box-shadow: 0 15px 35px -10px rgba(16, 185, 129, 0.2);
}

.result-card.fake {
    background: linear-gradient(135deg, rgba(136, 19, 55, 0.45) 0%, rgba(76, 5, 25, 0.75) 100%);
    border-color: rgba(244, 63, 94, 0.45);
    box-shadow: 0 15px 35px -10px rgba(244, 63, 94, 0.25);
}

.result-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: 0.02em;
}

.real .result-title { color: #34d399; }
.fake .result-title { color: #fb7185; }

.result-desc {
    color: #cbd5e1;
    font-size: 0.95rem;
    margin: 0.4rem 0 1.2rem;
}

.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.9rem;
}

.metric-box {
    background: rgba(10, 14, 23, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 0.9rem 1rem;
}

.metric-box small {
    display: block;
    color: #94a3b8;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
}

.metric-box b {
    color: #f8fafc;
    font-size: 1.35rem;
    font-family: 'Outfit', sans-serif;
}

/* Fusion Box */
.fusion-box {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin: 1rem 0;
    background: rgba(10, 14, 23, 0.5);
    padding: 0.9rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.fusion-item {
    background: rgba(15, 23, 42, 0.7);
    padding: 0.8rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    text-align: center;
}

.fusion-item small {
    color: #94a3b8;
    display: block;
    font-size: 0.78rem;
    margin-bottom: 0.2rem;
}

.fusion-item b {
    font-size: 1.15rem;
    color: #38bdf8;
    font-family: 'Outfit', sans-serif;
}

/* Empty State */
.empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    color: #64748b;
}

.empty-radar {
    width: 64px;
    height: 64px;
    margin: 0 auto 1.2rem;
    border-radius: 50%;
    background: rgba(56, 189, 248, 0.06);
    border: 1.5px dashed rgba(56, 189, 248, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
}

.empty-state h4 {
    font-family: 'Outfit', sans-serif;
    color: #94a3b8;
    font-size: 1.15rem;
    margin-bottom: 0.3rem;
}

.empty-state p {
    font-size: 0.88rem;
    max-width: 320px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Heatmap Container */
.heatmap-container {
    background: #0a0e17;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 1.2rem;
    margin-top: 1rem;
    text-align: center;
}

.heatmap-img {
    max-width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(56, 189, 248, 0.3);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.82rem;
    padding: 2.5rem 0 1rem;
}

/* DCT Analysis Panel */
.dct-panel {
    background: rgba(10, 14, 23, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}

.dct-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.dct-score-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0.7rem 0;
}

.dct-score-badge {
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
}

.dct-score-badge.low   { color: #34d399; }
.dct-score-badge.mid   { color: #fbbf24; }
.dct-score-badge.high  { color: #fb7185; }

.dct-note {
    color: #94a3b8;
    font-size: 0.83rem;
    line-height: 1.5;
    margin-top: 0.4rem;
}

.dct-features {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.8rem;
}

.dct-feat {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
}

.dct-feat small {
    display: block;
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.15rem;
}

.dct-feat b {
    color: #e2e8f0;
    font-size: 0.95rem;
    font-family: 'Outfit', sans-serif;
}

/* Model Scope Disclaimer */
.scope-note {
    background: rgba(251, 191, 36, 0.07);
    border: 1px solid rgba(251, 191, 36, 0.3);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #fbbf24;
    line-height: 1.55;
}

.scope-note b { color: #fde68a; }

@media(max-width: 840px) {
    .grid { grid-template-columns: 1fr; }
    .hero h1 { font-size: 2.2rem; }
    .dct-features { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("## 🛡️ DeepGuard")
st.sidebar.caption("Multimodal AI Media Forensics Engine")
st.sidebar.markdown("---")

modality = st.sidebar.radio("Inspection Modality", ["Image", "Audio", "Video"])

info = {
    "Image": "Analyzes facial landmarks, texture anomalies & Grad-CAM spatial heatmaps.",
    "Audio": "Analyzes speech frequencies using 40-band MFCC spectral forensics.",
    "Video": "Multi-frame spatial ensemble fused with automated acoustic track analysis.",
}
formats = {
    "Image": "JPG · JPEG · PNG · BMP",
    "Audio": "WAV · MP3 · M4A · FLAC",
    "Video": "MP4 · AVI · MOV · MKV",
}
st.sidebar.info(info[modality])
st.sidebar.markdown("**Supported File Types**")
st.sidebar.caption(formats[modality])

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Forensic Subsystems**")
st.sidebar.caption("🟢 Face ROI Localizer (Haar Cascade)")
st.sidebar.caption("🟢 Grad-CAM Spatial Heatmap Generator")
st.sidebar.caption("🟢 DCT Frequency-Domain AI Synthesis Detector")
st.sidebar.caption("🟢 FFmpeg 16kHz Acoustic Stream Extractor")
st.sidebar.caption("🟢 Multimodal Late-Fusion Decision Engine")

if modality == "Image":
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="scope-note">
        <b>⚠️ Model Scope Notice</b><br>
        The primary CNN was trained on <b>face-swap deepfakes</b> (DeepFaceLab / FaceSwap).
        It may not reliably detect images generated by <b>diffusion models</b>
        (Stable Diffusion, Midjourney, DALL-E, etc.).<br><br>
        The supplementary <b>DCT Spectral Detector</b> provides a secondary score
        specifically targeting AI-synthesis artifacts for all image types.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption("Engine: TensorFlow 2.21 / Keras 3")


# ----------------- MAIN HERO -----------------
st.markdown("""
<div class="hero">
    <div class="badge-row">
        <span class="badge-pill"><span class="pulse-dot"></span>AI FORENSIC VERIFICATION</span>
        <span class="badge-pill">V2.2 MULTIMODAL</span>
    </div>
    <h1>Multimodal Deepfake Detection</h1>
    <p>Empowered by deep convolutional networks, spatial Face ROI localization, Grad-CAM attention heatmaps, and multimodal video-audio fusion to deliver transparent, confidence-scored media authentication.</p>
</div>

<div class="grid">
    <div class="feature-card">
        <span class="f-icon">📸</span>
        <b>Face ROI & Grad-CAM</b>
        <span>Isolates facial geometry and generates spatial heatmaps of artifacts.</span>
    </div>
    <div class="feature-card">
        <span class="f-icon">🎙️</span>
        <b>Acoustic Spectral Analysis</b>
        <span>Evaluates MFCC spectrograms to detect synthetic voice cloning.</span>
    </div>
    <div class="feature-card">
        <span class="f-icon">🎥</span>
        <b>Audio-Visual Video Fusion</b>
        <span>Combines frame-level visual forensics with extracted speech tracks.</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------- INFERENCE HELPER -----------------
def predict_file(uploaded, mode):
    data = uploaded.getvalue()
    if mode == "Image":
        from image_model.image_predict import predict_image_bytes
        return predict_image_bytes(data, uploaded.name)
    if mode == "Audio":
        from audio_model.audio_predict import predict_audio_bytes
        return predict_audio_bytes(data, uploaded.name)
    from video_model.video_predict import predict_video_bytes
    return predict_video_bytes(data, uploaded.name)


# ----------------- RESULT RENDERER -----------------
def show_result(result, mode, file_name, file_size, file_hash):
    fake = bool(result.get("is_deepfake", False))
    conf = float(result.get("confidence", 0) or 0)
    prob = float(result.get("probability_fake", 0) or 0)

    # Check DCT result upfront so we can override the main verdict card
    dct_analysis = result.get("dct_analysis") if mode == "Image" else None
    dct_ai_prob = float(dct_analysis.get("ai_synthesis_probability") or 0) if dct_analysis else 0.0
    # Definitive AI verdict only at HIGH confidence (>=0.60) to avoid false positives
    dct_flags_ai = (not fake) and (dct_ai_prob >= 0.60)
    # Elevated suspicion (0.38-0.60): show warning but don't override main verdict
    dct_elevated  = (not fake) and (0.38 <= dct_ai_prob < 0.60)

    if dct_flags_ai:
        # Override: DCT says AI-generated even though CNN missed it
        cls = "fake"
        title = "🤖 AI GENERATED IMAGE DETECTED"
        desc = (
            f"Spectral frequency analysis detected AI-synthesis fingerprints in this image "
            f"({dct_ai_prob:.0%} AI probability). This image was generated by an AI tool such as "
            f"Stable Diffusion, Midjourney, DALL-E, or a similar generative model."
        )
    elif fake:
        cls = "fake"
        title = "⚠️ DEEPFAKE MANIPULATION DETECTED"
        desc = f"Forensic algorithms detected strong indicators of synthetic generation or manipulation in this {mode.lower()}."
    else:
        cls = "real"
        title = "✅ AUTHENTIC MEDIA VERIFIED"
        desc = f"No detectable adversarial artifacts or synthesis patterns found in this {mode.lower()}."

    # Pre-compute metric display values for the card
    metric1_label = "DCT AI Probability" if dct_flags_ai else "Model Confidence"
    metric1_value = f"{dct_ai_prob:.2%}" if dct_flags_ai else f"{conf:.2%}"
    metric2_value = f"{max(prob, dct_ai_prob):.2%}" if dct_flags_ai else f"{prob:.2%}"
    progress_val  = dct_ai_prob if dct_flags_ai else conf
    progress_label = f"{'AI Synthesis Certainty' if dct_flags_ai else 'Inference Certainty'}: {progress_val:.1%}"

    st.markdown(
        f"""
        <div class="result-card {cls}">
            <div class="result-title">{title}</div>
            <div class="result-desc">{desc}</div>
            <div class="metrics-grid">
                <div class="metric-box">
                    <small>{metric1_label}</small>
                    <b>{metric1_value}</b>
                </div>
                <div class="metric-box">
                    <small>Manipulation Probability</small>
                    <b>{metric2_value}</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(max(progress_val, 0.0), 1.0), text=progress_label)

    # NOTE: build_prediction_result() does payload.update(extra), so all extra
    # fields are at the TOP LEVEL of result (no nested 'extra' key).
    extra = result

    # Multimodal Video Fusion Panel
    if mode == "Video" and isinstance(extra, dict):
        if extra.get("multimodal_fusion_applied"):
            vis_p = extra.get("visual_probability_fake", 0.0)
            aud_p = extra.get("audio_probability_fake", 0.0)
            st.markdown(
                f"""
                <div class="fusion-box">
                    <div class="fusion-item">
                        <small>🎥 Visual Frame Score (60%)</small>
                        <b>{vis_p:.1%} Fake</b>
                    </div>
                    <div class="fusion-item">
                        <small>🎙️ Extracted Audio Score (40%)</small>
                        <b>{aud_p:.1%} Fake</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif extra.get("audio_detected") is False:
            st.caption("ℹ️ No audio channel found in video container; visual frame ensemble applied.")

        # Temporal Consistency Timeline
        per_frame = extra.get("per_frame_fake_probabilities")
        if per_frame and len(per_frame) > 1:
            st.markdown("##### 📈 Frame-by-Frame Manipulation Curve")
            df = pd.DataFrame({"Frame": range(1, len(per_frame) + 1), "Manipulation Probability": per_frame})
            st.line_chart(df.set_index("Frame"))

    # Grad-CAM Attention Heatmap
    if mode == "Image" and isinstance(extra, dict) and extra.get("heatmap_b64"):
        st.markdown("##### 🔬 Grad-CAM Spatial Explainability Heatmap")
        b64_data = extra["heatmap_b64"]
        st.markdown(
            f"""
            <div class="heatmap-container">
                <img src="data:image/png;base64,{b64_data}" class="heatmap-img" alt="Grad-CAM Overlay"/>
                <p style="color:#94a3b8; font-size:0.82rem; margin-top:0.6rem;">
                    {"👤 Face Region of Interest (ROI) isolated." if extra.get("face_detected") else "ℹ️ Standard crop applied."}
                    Warm/red areas indicate spatial pixel anomalies that influenced the prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # DCT Frequency-Domain AI Synthesis Panel
    if mode == "Image" and isinstance(extra, dict) and extra.get("dct_analysis"):
        dct = extra["dct_analysis"]
        ai_prob = dct.get("ai_synthesis_probability")
        if ai_prob is not None:
            # Updated thresholds matching 3-tier scoring in dct_detector.py
            if ai_prob >= 0.60:
                badge_cls, verdict_icon = "high", "🔴"
            elif ai_prob >= 0.38:
                badge_cls, verdict_icon = "mid", "🟡"
            else:
                badge_cls, verdict_icon = "low", "🟢"

            hf  = dct.get('hf_energy_ratio', 0)
            sf  = dct.get('spectral_flatness', 0)
            mp  = dct.get('mid_freq_peak_ratio', 0)
            bk  = dct.get('blocking_score', 0)
            nf  = dct.get('noise_floor_cv', 0)
            cc  = dct.get('channel_correlation', 0)
            av  = dct.get('azimuthal_variance', 0)
            note = dct.get('note', '')

            # Banner: definitive AI-generated OR elevated suspicion
            combined_banner = ""
            if dct_flags_ai:
                combined_banner = f"""
                <div style="
                    background: linear-gradient(135deg, rgba(239,68,68,0.18) 0%, rgba(220,38,38,0.12) 100%);
                    border: 1px solid rgba(239,68,68,0.5);
                    border-radius: 14px;
                    padding: 1rem 1.2rem;
                    margin-bottom: 0.9rem;
                ">
                    <div style="font-family:'Outfit',sans-serif; font-weight:800; color:#f87171; font-size:1.15rem; margin-bottom:0.4rem;">
                        🤖 AI Generated Image
                    </div>
                    <div style="color:#fca5a5; font-size:0.88rem; line-height:1.65;">
                        This image was <b>generated by an AI</b>. Frequency-domain spectral analysis
                        identified <b>{ai_prob:.0%} AI-synthesis probability</b> based on:
                        <ul style="margin:0.4rem 0 0.4rem 1.2rem; color:#fca5a5;">
                            <li>Absence of natural camera sensor noise in the high-frequency domain</li>
                            <li>Spectral flatness patterns consistent with AI neural network synthesis</li>
                            <li>RGB channel independence typical of generative models</li>
                        </ul>
                        Tools like <b>Stable Diffusion, Midjourney, DALL-E, Adobe Firefly</b> and similar AI
                        image generators produce these exact spectral signatures.
                    </div>
                </div>
                """
            elif dct_elevated:
                combined_banner = f"""
                <div style="
                    background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(234,88,12,0.08) 100%);
                    border: 1px solid rgba(245,158,11,0.40);
                    border-radius: 14px;
                    padding: 0.85rem 1.1rem;
                    margin-bottom: 0.9rem;
                ">
                    <div style="font-family:'Outfit',sans-serif; font-weight:800; color:#fbbf24; font-size:1.05rem; margin-bottom:0.35rem;">
                        ⚠️ Elevated AI-Synthesis Suspicion ({ai_prob:.0%})
                    </div>
                    <div style="color:#fde68a; font-size:0.85rem; line-height:1.6;">
                        Spectral analysis detected multiple AI-image markers. This image has a <b>high likelihood
                        of being AI-generated</b> (Stable Diffusion, Midjourney, DALL-E, etc.) even though the
                        confidence score did not reach the definitive threshold.
                    </div>
                </div>
                """

            st.markdown(
                f"""
                {combined_banner}
                <div class="dct-panel">
                    <div class="dct-title">📡 DCT Spectral Forensics — AI Synthesis Detector</div>
                    <div style="color:#64748b; font-size:0.8rem; margin-bottom:0.5rem;">
                        Secondary frequency-domain analysis · Noise-floor, spectral &amp; channel fingerprinting
                    </div>
                    <div class="dct-score-row">
                        <div class="dct-score-badge {badge_cls}">{verdict_icon} {ai_prob:.1%}</div>
                        <div style="color:#94a3b8; font-size:0.88rem;">
                            AI-Synthesis Probability<br>
                            <span style="font-size:0.78rem; color:#64748b;">Higher = more likely AI-generated</span>
                        </div>
                    </div>
                    <div class="dct-note">{note}</div>
                    <div class="dct-features" style="grid-template-columns: 1fr 1fr 1fr;">
                        <div class="dct-feat"><small>HF Energy Ratio</small><b>{hf:.4f}</b></div>
                        <div class="dct-feat"><small>Spectral Flatness</small><b>{sf:.4f}</b></div>
                        <div class="dct-feat"><small>Noise Floor CV</small><b>{nf:.4f}</b></div>
                        <div class="dct-feat"><small>Channel Correlation</small><b>{cc:.4f}</b></div>
                        <div class="dct-feat"><small>Azimuthal Variance</small><b>{av:.4f}</b></div>
                        <div class="dct-feat"><small>JPEG Blocking</small><b>{bk:.4f}</b></div>
                    </div>
                    <div style="color:#475569; font-size:0.75rem; margin-top:0.7rem;">
                        ⚠️ Heuristic detector — treat as supplementary signal, not ground truth.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"ℹ️ DCT analysis unavailable: {dct.get('note', 'unknown error')}")

    # Technical Expander
    with st.expander("🔎 Forensic Audit Parameters"):
        st.write(f"**Target Media:** `{file_name}`")
        st.write(
            f"**File Size:** {file_size/1024/1024:.2f} MB"
            if mode == "Video"
            else f"**File Size:** {file_size/1024:.2f} KB"
        )
        st.write(f"**SHA-256 Checksum:** `{file_hash}`")
        st.write(f"**Model Descriptor:** `{result.get('model_source', 'N/A')}`")
        _SKIP = {"heatmap_b64", "label_mapping", "raw_probabilities"}
        clean_extra = {
            k: ("[Encoded PNG Heatmap Stream]" if k == "heatmap_b64" else
                (f"[{len(v)} frames evaluated]" if k == "frame_probabilities" else v))
            for k, v in result.items() if k not in _SKIP
        }
        st.json(clean_extra)

    # Download Forensic Report
    # Exclude base prediction fields + binary heatmap from the JSON metadata
    _BASE_KEYS = {
        "modality", "label_mapping", "threshold", "raw_probabilities",
        "class_index", "label", "confidence", "is_deepfake",
        "probability_fake", "model_source", "heatmap_b64",
    }
    report_data = {
        "report_id": f"DG-{int(time.time())}",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "file_name": file_name,
        "file_size_bytes": file_size,
        "sha256_hash": file_hash,
        "modality": mode.lower(),
        "verdict": result.get("label", "UNKNOWN"),
        "is_deepfake": fake,
        "confidence": conf,
        "fake_probability": prob,
        "model_source": result.get("model_source", "N/A"),
        "forensic_metadata": {k: v for k, v in result.items() if k not in _BASE_KEYS},
    }
    st.download_button(
        label="📥 Download Forensic Audit Report (JSON)",
        data=json.dumps(report_data, indent=2),
        file_name=f"deepguard_audit_{mode.lower()}_{file_name}.json",
        mime="application/json",
    )


# ----------------- MAIN LAYOUT -----------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    icon = {"Image": "📸", "Audio": "🎙️", "Video": "🎥"}[modality]
    st.markdown(
        f"""
        <div class="glass-panel">
            <div class="panel-title">{icon} {modality} Ingestion & Scanner</div>
            <div class="panel-sub">Upload a {modality.lower()} file to initiate neural forensic evaluation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    types = {
        "Image": ["jpg", "jpeg", "png", "bmp"],
        "Audio": ["wav", "mp3", "m4a", "flac"],
        "Video": ["mp4", "avi", "mov", "mkv"],
    }
    uploaded = st.file_uploader(
        f"Upload {modality.lower()}",
        type=types[modality],
        key=f"{modality}_upload",
        label_visibility="collapsed",
    )

    if uploaded:
        if modality == "Image":
            st.image(uploaded, width="stretch")
        elif modality == "Audio":
            st.audio(uploaded)
        else:
            st.video(uploaded)
        st.caption(f"📄 **{uploaded.name}** · {uploaded.size/1024:.1f} KB")

        file_bytes = uploaded.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if st.button(f"🔍 Run Forensic Analysis ({modality})", key=f"{modality}_predict"):
            with st.spinner(f"Running {modality.lower()} forensic models..."):
                try:
                    st.session_state[f"{modality}_result"] = predict_file(uploaded, modality)
                    st.session_state[f"{modality}_name"] = uploaded.name
                    st.session_state[f"{modality}_size"] = uploaded.size
                    st.session_state[f"{modality}_hash"] = file_hash
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

with right:
    st.markdown(
        """
        <div class="glass-panel">
            <div class="panel-title">📊 Forensic Analysis Verdict</div>
            <div class="panel-sub">Neural model predictions and artifact diagnostics appear here.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.get(f"{modality}_result")
    if result:
        show_result(
            result,
            modality,
            st.session_state.get(f"{modality}_name", ""),
            st.session_state.get(f"{modality}_size", 0),
            st.session_state.get(f"{modality}_hash", "N/A"),
        )
    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-radar">📡</div>
                <h4>System Standby</h4>
                <p>Upload a media file on the left panel and click <b>Run Forensic Analysis</b> to inspect for synthetic manipulation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption("⚠️ DeepGuard predictions represent probabilistic forensic evaluations based on trained deep-learning models.")
st.markdown('<div class="footer">🛡️ DeepGuard · Multimodal Deepfake Forensics Engine · v2.2</div>', unsafe_allow_html=True)
