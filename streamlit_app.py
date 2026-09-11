import streamlit as st

st.set_page_config(page_title="DeepGuard | Deepfake Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#f7f9fc}.block-container{max-width:1180px;padding-top:2rem}
[data-testid="stSidebar"]{background:#101828}[data-testid="stSidebar"] *{color:#f2f4f7}
.hero{background:linear-gradient(135deg,#101828,#1d2939);padding:2.5rem 2.7rem;border-radius:24px;margin-bottom:1.4rem;box-shadow:0 12px 35px rgba(16,24,40,.12)}
.badge{display:inline-block;padding:.35rem .75rem;border-radius:99px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.12);color:#d0d5dd;font-size:.75rem;font-weight:700;letter-spacing:.04em}
.hero h1{color:white;font-size:2.6rem;line-height:1.1;margin:.75rem 0 .6rem;font-weight:800}.hero p{color:#d0d5dd;max-width:760px;font-size:1rem;line-height:1.6;margin:0}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin:1rem 0 1.5rem}.feature{background:#fff;border:1px solid #e4e7ec;border-radius:16px;padding:1rem}.feature b{display:block;color:#101828;margin-top:.3rem}.feature span{color:#667085;font-size:.78rem}
.card{background:#fff;border:1px solid #e4e7ec;border-radius:18px;padding:1.35rem;box-shadow:0 5px 18px rgba(16,24,40,.05);margin-bottom:1rem}.title{font-size:1.15rem;font-weight:750;color:#101828}.sub{color:#667085;font-size:.88rem;margin:.3rem 0 1rem}
[data-testid="stFileUploader"]{background:#f8fafc;border:1.5px dashed #98a2b3;border-radius:16px;padding:.7rem}[data-testid="stFileUploader"] section{border:0;background:transparent}
.stButton>button{width:100%;min-height:48px;border:0;border-radius:12px;background:#111827;color:white;font-weight:750;box-shadow:0 5px 14px rgba(17,24,39,.16)}.stButton>button:hover{background:#1f2937;color:white}
.result{border-radius:18px;padding:1.4rem;border:1px solid;margin-bottom:1rem}.real{background:#ecfdf3;border-color:#abefc6}.fake{background:#fef3f2;border-color:#fecdca}.rtitle{font-size:1.3rem;font-weight:850}.real .rtitle{color:#027a48}.fake .rtitle{color:#b42318}.desc{color:#475467;font-size:.9rem;margin:.4rem 0 1rem}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}.metric{background:rgba(255,255,255,.75);border-radius:12px;padding:.8rem}.metric small{display:block;color:#667085}.metric b{color:#101828;font-size:1.05rem}
.footer{text-align:center;color:#98a2b3;font-size:.78rem;padding:1.5rem 0}.stProgress>div>div>div>div{border-radius:99px}
@media(max-width:800px){.grid{grid-template-columns:1fr}.hero h1{font-size:2rem}}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("# 🛡️ DeepGuard")
st.sidebar.caption("Multimodal Deepfake Detection")
st.sidebar.markdown("---")
modality=st.sidebar.radio("Detection modality",["Image","Audio","Video"])
info={"Image":"Analyze images for signs of manipulation.","Audio":"Analyze speech/audio for synthetic content.","Video":"Sample video frames for deepfake indicators."}
formats={"Image":"JPG · JPEG · PNG · BMP","Audio":"WAV · MP3 · M4A · FLAC","Video":"MP4 · AVI · MOV · MKV"}
st.sidebar.info(info[modality]);st.sidebar.markdown("**Supported formats**");st.sidebar.caption(formats[modality]);st.sidebar.markdown("---");st.sidebar.caption("Powered by TensorFlow / Keras")

st.markdown("""
<div class="hero"><span class="badge">AI-POWERED MEDIA FORENSICS</span><h1>Multimodal Deepfake Detection</h1><p>Detect manipulated content across images, audio, and video using trained neural-network models. Upload your media and receive a clear, confidence-based prediction.</p></div>
<div class="grid"><div class="feature">📸<b>Image Analysis</b><span>Detect manipulated facial imagery.</span></div><div class="feature">🎙️<b>Audio Analysis</b><span>Inspect synthetic or altered speech.</span></div><div class="feature">🎥<b>Video Analysis</b><span>Evaluate sampled video frames.</span></div></div>
""", unsafe_allow_html=True)

def predict_file(uploaded, mode):
    data=uploaded.getvalue()
    if mode=="Image":
        from image_model.image_predict import predict_image_bytes
        return predict_image_bytes(data,uploaded.name)
    if mode=="Audio":
        from audio_model.audio_predict import predict_audio_bytes
        return predict_audio_bytes(data,uploaded.name)
    from video_model.video_predict import predict_video_bytes
    return predict_video_bytes(data,uploaded.name)

def show_result(result,mode,file_name,file_size):
    fake=bool(result.get("is_deepfake",False));conf=float(result.get("confidence",0) or 0);prob=float(result.get("probability_fake",0) or 0)
    cls="fake" if fake else "real";title="⚠️ DEEPFAKE DETECTED" if fake else "✅ AUTHENTIC MEDIA"
    desc=f"The model classified this {mode.lower()} as potentially manipulated." if fake else f"The model classified this {mode.lower()} as likely authentic."
    st.markdown(f'<div class="result {cls}"><div class="rtitle">{title}</div><div class="desc">{desc}</div><div class="metrics"><div class="metric"><small>Model confidence</small><b>{conf:.2%}</b></div><div class="metric"><small>Fake probability</small><b>{prob:.2%}</b></div></div></div>',unsafe_allow_html=True)
    st.progress(min(max(conf,0),1),text=f"Confidence · {conf:.1%}")
    with st.expander("🔎 View technical details"):
        st.write(f"**File:** {file_name}")
        st.write(f"**File size:** {file_size/1024/1024:.2f} MB" if mode=="Video" else f"**File size:** {file_size/1024:.2f} KB")
        st.write(f"**Model source:** {result.get('model_source','N/A')}")
        if "extra" in result:
            extra=result["extra"].copy() if isinstance(result["extra"],dict) else result["extra"]
            if isinstance(extra,dict) and "frame_probabilities" in extra: extra["frame_probabilities"]=f"[{len(extra['frame_probabilities'])} frames]"
            st.json(extra)

left,right=st.columns([1.05,.95],gap="large")
with left:
    icon={"Image":"📸","Audio":"🎙️","Video":"🎥"}[modality]
    st.markdown(f'<div class="card"><div class="title">{icon} {modality} Detection</div><div class="sub">Upload a {modality.lower()} file to begin analysis.</div></div>',unsafe_allow_html=True)
    types={"Image":["jpg","jpeg","png","bmp"],"Audio":["wav","mp3","m4a","flac"],"Video":["mp4","avi","mov","mkv"]}
    uploaded=st.file_uploader(f"Upload {modality.lower()}",type=types[modality],key=f"{modality}_upload",label_visibility="collapsed")
    if uploaded:
        if modality=="Image": st.image(uploaded,use_container_width=True)
        elif modality=="Audio": st.audio(uploaded)
        else: st.video(uploaded)
        st.caption(f"📄 {uploaded.name} · {uploaded.size/1024:.1f} KB")
        if st.button(f"🔍 Analyze {modality}",key=f"{modality}_predict"):
            with st.spinner(f"Analyzing {modality.lower()}... Please wait."):
                try:
                    st.session_state[f"{modality}_result"]=predict_file(uploaded,modality)
                    st.session_state[f"{modality}_name"]=uploaded.name
                    st.session_state[f"{modality}_size"]=uploaded.size
                except Exception as e: st.error(f"Analysis failed: {e}")
with right:
    st.markdown('<div class="card"><div class="title">📊 Analysis Result</div><div class="sub">Your prediction will appear here after analysis.</div></div>',unsafe_allow_html=True)
    result=st.session_state.get(f"{modality}_result")
    if result: show_result(result,modality,st.session_state.get(f"{modality}_name",""),st.session_state.get(f"{modality}_size",0))
    else: st.info("👆 Upload a file on the left and click **Analyze** to receive a prediction.")

st.markdown("---")
st.caption("⚠️ Detection results are model predictions and should not be treated as definitive proof of authenticity.")
st.markdown('<div class="footer">🛡️ DeepGuard · Multimodal Deepfake Detection System · v2.0</div>',unsafe_allow_html=True)
