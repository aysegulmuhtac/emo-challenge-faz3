import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
import os
import gdown
import time
 
st.set_page_config(
    page_title="Emo-Challenge 2026",
    page_icon="🧠",
    layout="wide"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
 
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
 
.stApp {
    background: #050816;
    color: white;
}
 
/* ── HERO ── */
.hero {
    text-align: center;
    padding: 60px 20px 40px;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 50px;
    padding: 6px 20px;
    font-size: 13px;
    color: #a78bfa;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 56px;
    font-weight: 900;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0 0 16px 0;
}
.hero-sub {
    font-size: 18px;
    color: rgba(255,255,255,0.5);
    max-width: 500px;
    margin: 0 auto 30px;
    line-height: 1.6;
}
.hero-team {
    font-size: 13px;
    color: rgba(255,255,255,0.3);
    letter-spacing: 1px;
}
 
/* ── GLASS CARD ── */
.glass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 32px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
}
 
/* ── UPLOAD AREA ── */
.upload-zone {
    background: rgba(139,92,246,0.06);
    border: 2px dashed rgba(139,92,246,0.35);
    border-radius: 16px;
    padding: 40px 20px;
    text-align: center;
    transition: all 0.3s;
    cursor: pointer;
}
.upload-zone:hover {
    background: rgba(139,92,246,0.12);
    border-color: rgba(139,92,246,0.6);
}
.upload-icon { font-size: 48px; margin-bottom: 12px; }
.upload-title { font-size: 18px; font-weight: 600; color: white; margin-bottom: 6px; }
.upload-sub { font-size: 13px; color: rgba(255,255,255,0.4); }
.format-tags { margin-top: 14px; }
.format-tag {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    color: rgba(255,255,255,0.5);
    margin: 3px;
}
 
/* ── RESULT CARD ── */
.result-card {
    border-radius: 24px;
    padding: 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.6s ease;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-emoji-big { font-size: 80px; display: block; margin-bottom: 8px; }
.result-emotion {
    font-size: 44px;
    font-weight: 900;
    color: white;
    letter-spacing: -1px;
    text-transform: uppercase;
    margin: 0;
}
.result-conf {
    font-size: 16px;
    color: rgba(255,255,255,0.7);
    margin-top: 8px;
}
 
/* ── PROB BARS ── */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 14px;
    gap: 12px;
}
.prob-label-txt {
    font-size: 14px;
    font-weight: 600;
    color: rgba(255,255,255,0.85);
    width: 90px;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 1s ease;
}
.prob-pct {
    font-size: 13px;
    font-weight: 700;
    color: rgba(255,255,255,0.7);
    width: 45px;
    text-align: right;
    flex-shrink: 0;
}
 
/* ── STATS ── */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}
.stat-val {
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-lbl {
    font-size: 12px;
    color: rgba(255,255,255,0.4);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
 
/* ── SECTION TITLE ── */
.sec-title {
    font-size: 13px;
    font-weight: 700;
    color: rgba(255,255,255,0.35);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 32px 0 16px;
}
 
/* ── DIVIDER ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent);
    margin: 30px 0;
}
 
/* ── FOOTER ── */
.footer-txt {
    text-align: center;
    color: rgba(255,255,255,0.2);
    font-size: 12px;
    padding: 30px 0 10px;
}
 
/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
 
div[data-testid="stFileUploader"] label { display: none; }
div[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)
 
# ── Duygu ayarları ─────────────────────────────────────────────
EMOJI  = {'Furious':'😡','Happy':'😊','Neutral':'😐','Sad':'😢','Shocked':'😲'}
GRADIENT = {
    'Furious': 'linear-gradient(135deg, #1a0505 0%, #3d0a0a 100%)',
    'Happy':   'linear-gradient(135deg, #1a1505 0%, #3d3000 100%)',
    'Neutral': 'linear-gradient(135deg, #0a0a1a 0%, #0f0f2e 100%)',
    'Sad':     'linear-gradient(135deg, #05051a 0%, #0a0a3d 100%)',
    'Shocked': 'linear-gradient(135deg, #12051a 0%, #280a3d 100%)',
}
BAR_COLOR = {
    'Furious': 'linear-gradient(90deg,#ff416c,#ff4b2b)',
    'Happy':   'linear-gradient(90deg,#f7971e,#ffd200)',
    'Neutral': 'linear-gradient(90deg,#8e9eab,#eef2f3)',
    'Sad':     'linear-gradient(90deg,#4776e6,#8e54e9)',
    'Shocked': 'linear-gradient(90deg,#a18cd1,#fbc2eb)',
}
GLOW = {
    'Furious':'rgba(255,65,108,0.3)',
    'Happy':'rgba(247,151,30,0.3)',
    'Neutral':'rgba(142,158,171,0.3)',
    'Sad':'rgba(71,118,230,0.3)',
    'Shocked':'rgba(161,140,209,0.3)',
}
 
# ── Model yükle ────────────────────────────────────────────────
MODEL_DIR = 'models'
@st.cache_resource
def load_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    files = {
        'model.pkl':         '1J7Vby6G15LQQyB4FJIgzjJxRYM_YSRvf',
        'scaler.pkl':        '1HWQFU3xvzkulf-wructZl_SCEi4ToZTn',
        'selector.pkl':      '160NjaN3RqBEFZ25vjaSLxuzeYDgDlunz',
        'label_encoder.pkl': '1aw2DpKfn-J9l_TJqqCNxqWxn0L0RwS0o'
    }
    for fname, fid in files.items():
        path = f'{MODEL_DIR}/{fname}'
        if not os.path.exists(path):
            gdown.download(f'https://drive.google.com/uc?id={fid}', path, quiet=True)
    with open(f'{MODEL_DIR}/model.pkl','rb') as f: model = pickle.load(f)
    with open(f'{MODEL_DIR}/scaler.pkl','rb') as f: scaler = pickle.load(f)
    with open(f'{MODEL_DIR}/selector.pkl','rb') as f: selector = pickle.load(f)
    with open(f'{MODEL_DIR}/label_encoder.pkl','rb') as f: le = pickle.load(f)
    return model, scaler, selector, le
 
# ── Öznitelik çıkarma ─────────────────────────────────────────
def extract_features(y, sr):
    try:
        f = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
        f.extend(np.mean(mfcc,axis=1)); f.extend(np.std(mfcc,axis=1))
        d1 = librosa.feature.delta(mfcc)
        f.extend(np.mean(d1,axis=1)); f.extend(np.std(d1,axis=1))
        d2 = librosa.feature.delta(mfcc,order=2)
        f.extend(np.mean(d2,axis=1)); f.extend(np.std(d2,axis=1))
        zcr = librosa.feature.zero_crossing_rate(y)
        f.append(np.mean(zcr)); f.append(np.std(zcr))
        rms = librosa.feature.rms(y=y)
        f.append(np.mean(rms)); f.append(np.std(rms))
        sc = librosa.feature.spectral_centroid(y=y,sr=sr)
        f.append(np.mean(sc)); f.append(np.std(sc))
        bw = librosa.feature.spectral_bandwidth(y=y,sr=sr)
        f.append(np.mean(bw)); f.append(np.std(bw))
        ro = librosa.feature.spectral_rolloff(y=y,sr=sr)
        f.append(np.mean(ro)); f.append(np.std(ro))
        co = librosa.feature.spectral_contrast(y=y,sr=sr,n_bands=6)
        f.extend(np.mean(co,axis=1)); f.extend(np.std(co,axis=1))
        ch = librosa.feature.chroma_stft(y=y,sr=sr)
        f.extend(np.mean(ch,axis=1)); f.extend(np.std(ch,axis=1))
        mel = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=64)
        mel_db = librosa.power_to_db(mel,ref=np.max)
        f.append(np.mean(mel_db)); f.append(np.std(mel_db))
        f.append(np.max(mel_db));  f.append(np.min(mel_db))
        try:
            yh = librosa.effects.harmonic(y)
            tz = librosa.feature.tonnetz(y=yh,sr=sr)
            f.extend(np.mean(tz,axis=1)); f.extend(np.std(tz,axis=1))
        except:
            f.extend([0]*12)
        return np.array(f, dtype=np.float32)
    except:
        return None
 
def predict(path, model, scaler, selector, le):
    y, sr = librosa.load(path, duration=3, offset=0.5)
    feat = extract_features(y, sr)
    if feat is None: return None, None, None
    t0 = time.time()
    fs = scaler.transform([feat])
    fp = selector.transform(fs)
    pred = model.predict(fp)[0]
    proba = model.predict_proba(fp)[0]
    inf_time = time.time() - t0
    emotion = le.inverse_transform([pred])[0]
    return emotion, dict(zip(le.classes_, proba)), inf_time
 
# ═══════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════
 
# HERO
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧠 &nbsp; Emo-Challenge 2026</div>
    <h1 class="hero-title">Real-Time Emotion<br>Analysis System</h1>
    <p class="hero-sub">Upload an audio file and let the AI detect the emotion in the voice with high accuracy.</p>
    <p class="hero-team">GROUP 12 &nbsp;·&nbsp; Ayşegül Muhtaç &nbsp;·&nbsp; Sena Poyraz &nbsp;·&nbsp; Yiğit Kadir Gökdemir</p>
</div>
""", unsafe_allow_html=True)
 
# STATS ROW
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-val">94.37%</div><div class="stat-lbl">Accuracy</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-val">424</div><div class="stat-lbl">Features</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-val">3,375</div><div class="stat-lbl">Training Samples</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-val">5</div><div class="stat-lbl">Emotion Classes</div></div>', unsafe_allow_html=True)
 
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
 
# Model yükle (sessiz)
with st.spinner('Loading AI model...'):
    try:
        model, scaler, selector, le = load_models()
    except Exception as e:
        st.error(f'Model could not be loaded: {e}')
        st.stop()
 
# UPLOAD SECTION
st.markdown('<div class="sec-title">🎙️ Audio Input</div>', unsafe_allow_html=True)
 
st.markdown("""
<div class="upload-zone">
    <div class="upload-icon">📂</div>
    <div class="upload-title">Drag & Drop Audio File</div>
    <div class="upload-sub">or click below to browse your files</div>
    <div class="format-tags">
        <span class="format-tag">WAV</span>
        <span class="format-tag">MP3</span>
        <span class="format-tag">FLAC</span>
    </div>
</div>
""", unsafe_allow_html=True)
 
uploaded = st.file_uploader("", type=['wav','mp3','flac'])
 
if uploaded:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Waveform + audio player
    st.markdown('<div class="sec-title">🎵 Audio Preview</div>', unsafe_allow_html=True)
    st.audio(uploaded, format='audio/wav')
 
    # Analiz
    with st.spinner('⚡ Analyzing emotion...'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        emotion, probs, inf_time = predict(tmp_path, model, scaler, selector, le)
        os.unlink(tmp_path)
 
    if emotion:
        emoji    = EMOJI.get(emotion,'🎭')
        grad     = GRADIENT.get(emotion,'linear-gradient(135deg,#1a1a2e,#16213e)')
        glow     = GLOW.get(emotion,'rgba(139,92,246,0.3)')
        conf     = probs[emotion]*100
 
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">🎯 Emotion Prediction Result</div>', unsafe_allow_html=True)
 
        # Result card
        st.markdown(f"""
        <div class="result-card" style="background:{grad}; box-shadow: 0 0 60px {glow}, 0 25px 50px rgba(0,0,0,0.5); border: 1px solid {glow};">
            <span class="result-emoji-big">{emoji}</span>
            <p class="result-emotion">{emotion}</p>
            <p class="result-conf">Confidence Score &nbsp;·&nbsp; <strong style="color:white">{conf:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
 
        # Circular-style confidence + inference info
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{conf:.1f}%</div><div class="stat-lbl">Confidence</div></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{inf_time:.2f}s</div><div class="stat-lbl">Inference Time</div></div>', unsafe_allow_html=True)
        with col_c:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{uploaded.name[:12]}</div><div class="stat-lbl">File</div></div>', unsafe_allow_html=True)
 
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">📊 Emotion Distribution</div>', unsafe_allow_html=True)
 
        # Probability bars
        bars_html = ""
        for cls, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            e = EMOJI.get(cls,'🎭')
            pct = p * 100
            bar_col = BAR_COLOR.get(cls,'linear-gradient(90deg,#667eea,#764ba2)')
            bars_html += f"""
            <div class="prob-row">
                <div class="prob-label-txt">{e} {cls}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{bar_col}"></div>
                </div>
                <div class="prob-pct">{pct:.1f}%</div>
            </div>
            """
        st.markdown(f'<div class="glass">{bars_html}</div>', unsafe_allow_html=True)
 
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">🤖 Model Performance</div>', unsafe_allow_html=True)
 
        m1,m2,m3,m4 = st.columns(4)
        with m1:
            st.markdown('<div class="stat-card"><div class="stat-val">94.37%</div><div class="stat-lbl">Test Accuracy</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="stat-card"><div class="stat-val">Ensemble</div><div class="stat-lbl">Model Type</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="stat-card"><div class="stat-val">MFCC+Δ</div><div class="stat-lbl">Features</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown('<div class="stat-card"><div class="stat-val">4th</div><div class="stat-lbl">Leaderboard</div></div>', unsafe_allow_html=True)
 
    else:
        st.error('❌ Could not analyze audio. Please try a different file.')
 
st.markdown("""
<div class="footer-txt">
    BIL216 Signals and Systems &nbsp;·&nbsp; Emo-Challenge 2026 Phase 3<br>
    Voting Ensemble: 2×SVM + LightGBM + XGBoost + MLP &nbsp;·&nbsp; 424 Features &nbsp;·&nbsp; Data Augmentation (×5)
</div>
""", unsafe_allow_html=True)
 
