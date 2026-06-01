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
 
* { font-family: 'Inter', sans-serif; box-sizing: border-box; margin: 0; padding: 0; }
 
.stApp {
    background: #F0F4FA;
    color: #0f172a;
}
 
/* ── TOPBAR ── */
.topbar {
    background: white;
    border-bottom: 1px solid #e2e8f0;
    padding: 16px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0;
}
.topbar-logo {
    font-size: 18px;
    font-weight: 800;
    color: #1e293b;
    letter-spacing: -0.5px;
}
.topbar-logo span {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.topbar-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 50px;
    padding: 6px 18px;
    font-size: 13px;
    font-weight: 600;
}
 
/* ── HERO ── */
.hero {
    background: white;
    border-bottom: 1px solid #e2e8f0;
    padding: 56px 40px 48px;
    text-align: center;
}
.hero-eyebrow {
    font-size: 13px;
    font-weight: 700;
    color: #6366f1;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero-title {
    font-size: 52px;
    font-weight: 900;
    color: #0f172a;
    line-height: 1.1;
    letter-spacing: -2px;
    margin-bottom: 16px;
}
.hero-title span {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 18px;
    color: #64748b;
    max-width: 520px;
    margin: 0 auto 24px;
    line-height: 1.7;
    font-weight: 400;
}
.hero-team {
    font-size: 14px;
    color: #94a3b8;
    font-weight: 500;
    letter-spacing: 0.5px;
}
 
/* ── STATS ── */
.stats-row {
    background: white;
    border-bottom: 1px solid #e2e8f0;
    padding: 24px 40px;
    display: flex;
    justify-content: center;
    gap: 60px;
}
.stat-item { text-align: center; }
.stat-val {
    font-size: 32px;
    font-weight: 900;
    color: #0f172a;
    letter-spacing: -1px;
}
.stat-val span {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-lbl {
    font-size: 12px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 2px;
}
 
/* ── CONTENT ── */
.content { padding: 40px; max-width: 1100px; margin: 0 auto; }
 
/* ── CARD ── */
.card {
    background: white;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    padding: 36px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 24px;
}
.card-title {
    font-size: 13px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 20px;
}
 
/* ── UPLOAD ZONE ── */
.upload-zone {
    background: #fafbff;
    border: 2px dashed #c7d2fe;
    border-radius: 16px;
    padding: 48px 24px;
    text-align: center;
    transition: all 0.2s;
}
.upload-icon { font-size: 52px; margin-bottom: 14px; }
.upload-title { font-size: 20px; font-weight: 700; color: #1e293b; margin-bottom: 8px; }
.upload-sub { font-size: 15px; color: #94a3b8; margin-bottom: 16px; }
.format-tag {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    margin: 3px;
}
 
/* ── RESULT CARD ── */
.result-card {
    border-radius: 20px;
    padding: 48px 36px;
    text-align: center;
    border: 1px solid;
    animation: fadeUp 0.5s ease;
    margin-bottom: 24px;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(24px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-emoji { font-size: 88px; display:block; margin-bottom: 12px; }
.result-emotion {
    font-size: 52px;
    font-weight: 900;
    letter-spacing: -2px;
    margin-bottom: 8px;
}
.result-conf { font-size: 18px; font-weight: 500; }
 
/* ── PROB BAR ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 16px;
}
.prob-emoji { font-size: 22px; width: 30px; }
.prob-name { font-size: 16px; font-weight: 600; color: #1e293b; width: 90px; }
.prob-bg {
    flex: 1;
    background: #f1f5f9;
    border-radius: 12px;
    height: 12px;
    overflow: hidden;
}
.prob-fill { height:100%; border-radius:12px; transition: width 0.8s ease; }
.prob-pct { font-size: 15px; font-weight: 700; color: #475569; width: 50px; text-align:right; }
 
/* ── PERF CARD ── */
.perf-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}
.perf-item {
    background: #fafbff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
}
.perf-val {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.perf-lbl {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
 
/* ── INFERENCE INFO ── */
.inf-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.inf-item {
    background: #fafbff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}
.inf-val { font-size: 24px; font-weight: 800; color: #0f172a; }
.inf-lbl { font-size: 11px; font-weight: 700; color: #94a3b8; text-transform:uppercase; letter-spacing:1.5px; margin-top:4px; }
 
/* ── FOOTER ── */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    padding: 32px 0 16px;
    border-top: 1px solid #e2e8f0;
    margin-top: 20px;
    line-height: 1.8;
}
 
/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display:none; }
div[data-testid="stFileUploader"] label { display:none; }
div[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)
 
# ── Config ──────────────────────────────────────────────────────
EMOJI = {'Furious':'😡','Happy':'😊','Neutral':'😐','Sad':'😢','Shocked':'😲'}
 
EMOTION_STYLE = {
    'Furious': {'bg':'#fff5f5', 'border':'#fca5a5', 'text':'#b91c1c', 'bar':'linear-gradient(90deg,#ef4444,#dc2626)'},
    'Happy':   {'bg':'#fffbeb', 'border':'#fcd34d', 'text':'#b45309', 'bar':'linear-gradient(90deg,#f59e0b,#d97706)'},
    'Neutral': {'bg':'#f8fafc', 'border':'#cbd5e1', 'text':'#475569', 'bar':'linear-gradient(90deg,#94a3b8,#64748b)'},
    'Sad':     {'bg':'#eff6ff', 'border':'#93c5fd', 'text':'#1d4ed8', 'bar':'linear-gradient(90deg,#3b82f6,#2563eb)'},
    'Shocked': {'bg':'#faf5ff', 'border':'#c4b5fd', 'text':'#7c3aed', 'bar':'linear-gradient(90deg,#8b5cf6,#7c3aed)'},
}
 
DEFAULT_BAR = 'linear-gradient(90deg,#6366f1,#8b5cf6)'
 
# ── Model ────────────────────────────────────────────────────────
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
    with open(f'{MODEL_DIR}/model.pkl','rb') as f: model=pickle.load(f)
    with open(f'{MODEL_DIR}/scaler.pkl','rb') as f: scaler=pickle.load(f)
    with open(f'{MODEL_DIR}/selector.pkl','rb') as f: selector=pickle.load(f)
    with open(f'{MODEL_DIR}/label_encoder.pkl','rb') as f: le=pickle.load(f)
    return model, scaler, selector, le
 
def extract_features(y, sr):
    try:
        f=[]
        mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=60)
        f.extend(np.mean(mfcc,axis=1)); f.extend(np.std(mfcc,axis=1))
        d1=librosa.feature.delta(mfcc)
        f.extend(np.mean(d1,axis=1)); f.extend(np.std(d1,axis=1))
        d2=librosa.feature.delta(mfcc,order=2)
        f.extend(np.mean(d2,axis=1)); f.extend(np.std(d2,axis=1))
        zcr=librosa.feature.zero_crossing_rate(y)
        f.append(np.mean(zcr)); f.append(np.std(zcr))
        rms=librosa.feature.rms(y=y)
        f.append(np.mean(rms)); f.append(np.std(rms))
        sc=librosa.feature.spectral_centroid(y=y,sr=sr)
        f.append(np.mean(sc)); f.append(np.std(sc))
        bw=librosa.feature.spectral_bandwidth(y=y,sr=sr)
        f.append(np.mean(bw)); f.append(np.std(bw))
        ro=librosa.feature.spectral_rolloff(y=y,sr=sr)
        f.append(np.mean(ro)); f.append(np.std(ro))
        co=librosa.feature.spectral_contrast(y=y,sr=sr,n_bands=6)
        f.extend(np.mean(co,axis=1)); f.extend(np.std(co,axis=1))
        ch=librosa.feature.chroma_stft(y=y,sr=sr)
        f.extend(np.mean(ch,axis=1)); f.extend(np.std(ch,axis=1))
        mel=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=64)
        mel_db=librosa.power_to_db(mel,ref=np.max)
        f.append(np.mean(mel_db)); f.append(np.std(mel_db))
        f.append(np.max(mel_db)); f.append(np.min(mel_db))
        try:
            yh=librosa.effects.harmonic(y)
            tz=librosa.feature.tonnetz(y=yh,sr=sr)
            f.extend(np.mean(tz,axis=1)); f.extend(np.std(tz,axis=1))
        except:
            f.extend([0]*12)
        return np.array(f,dtype=np.float32)
    except:
        return None
 
def predict(path, model, scaler, selector, le):
    y,sr=librosa.load(path,duration=3,offset=0.5)
    feat=extract_features(y,sr)
    if feat is None: return None,None,None
    t0=time.time()
    fs=scaler.transform([feat])
    fp=selector.transform(fs)
    pred=model.predict(fp)[0]
    proba=model.predict_proba(fp)[0]
    inf=time.time()-t0
    emotion=le.inverse_transform([pred])[0]
    return emotion,dict(zip(le.classes_,proba)),inf
 
# ═══════════════════════════════ UI ════════════════════════════
 
# TOPBAR
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">🧠 &nbsp;<span>Emo</span>-Challenge 2026</div>
    <div class="topbar-badge">Phase 3 — Live Demo</div>
</div>
""", unsafe_allow_html=True)
 
# HERO
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI · Speech · Emotion Recognition</div>
    <h1 class="hero-title">Real-Time <span>Emotion</span><br>Analysis System</h1>
    <p class="hero-sub">Upload an audio file. The AI model analyzes acoustic features and detects the emotion in the voice.</p>
    <p class="hero-team">Group 12 &nbsp;·&nbsp; Ayşegül Muhtaç &nbsp;·&nbsp; Sena Poyraz &nbsp;·&nbsp; Yiğit Kadir Gökdemir</p>
</div>
""", unsafe_allow_html=True)
 
# STATS
st.markdown("""
<div class="stats-row">
    <div class="stat-item">
        <div class="stat-val"><span>94.37%</span></div>
        <div class="stat-lbl">Model Accuracy</div>
    </div>
    <div class="stat-item">
        <div class="stat-val"><span>424</span></div>
        <div class="stat-lbl">Audio Features</div>
    </div>
    <div class="stat-item">
        <div class="stat-val"><span>3,375</span></div>
        <div class="stat-lbl">Training Samples</div>
    </div>
    <div class="stat-item">
        <div class="stat-val"><span>5</span></div>
        <div class="stat-lbl">Emotion Classes</div>
    </div>
    <div class="stat-item">
        <div class="stat-val"><span>4th</span></div>
        <div class="stat-lbl">Leaderboard Rank</div>
    </div>
</div>
""", unsafe_allow_html=True)
 
# Model yükle
with st.spinner('Loading model...'):
    try:
        model, scaler, selector, le = load_models()
    except Exception as e:
        st.error(f'Model could not be loaded: {e}')
        st.stop()
 
# UPLOAD
st.markdown('<div style="max-width:900px;margin:40px auto;padding:0 20px">', unsafe_allow_html=True)
 
st.markdown("""
<div class="card">
    <div class="card-title">🎙️ Audio Input</div>
    <div class="upload-zone">
        <div class="upload-icon">📂</div>
        <div class="upload-title">Drag & Drop Your Audio File</div>
        <div class="upload-sub">or use the button below to browse files</div>
        <div style="margin-top:12px">
            <span class="format-tag">WAV</span>
            <span class="format-tag">MP3</span>
            <span class="format-tag">FLAC</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
 
uploaded = st.file_uploader("", type=['wav','mp3','flac'])
 
if uploaded:
    # Audio player
    st.markdown("""
    <div class="card">
        <div class="card-title">🎵 Audio Preview</div>
    </div>
    """, unsafe_allow_html=True)
    st.audio(uploaded, format='audio/wav')
 
    # Analiz
    with st.spinner('⚡ Analyzing emotion...'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        emotion, probs, inf_time = predict(tmp_path, model, scaler, selector, le)
        os.unlink(tmp_path)
 
    if emotion:
        style = EMOTION_STYLE.get(emotion, {'bg':'#fafbff','border':'#e2e8f0','text':'#0f172a','bar':DEFAULT_BAR})
        emoji = EMOJI.get(emotion, '🎭')
        conf  = probs[emotion]*100
 
        # RESULT CARD
        st.markdown(f"""
        <div class="result-card" style="
            background:{style['bg']};
            border-color:{style['border']};
            box-shadow: 0 0 0 4px {style['border']}44, 0 20px 40px rgba(0,0,0,0.08);
        ">
            <span class="result-emoji">{emoji}</span>
            <div class="result-emotion" style="color:{style['text']}">{emotion}</div>
            <div class="result-conf" style="color:{style['text']}99">
                Confidence Score &nbsp;·&nbsp; <strong style="color:{style['text']}">{conf:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        # INFERENCE INFO
        st.markdown(f"""
        <div class="inf-row">
            <div class="inf-item">
                <div class="inf-val" style="color:{style['text']}">{conf:.1f}%</div>
                <div class="inf-lbl">Confidence</div>
            </div>
            <div class="inf-item">
                <div class="inf-val">{inf_time:.2f}s</div>
                <div class="inf-lbl">Inference Time</div>
            </div>
            <div class="inf-item">
                <div class="inf-val" style="font-size:16px">{uploaded.name[:16]}</div>
                <div class="inf-lbl">File Name</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        # PROB BARS
        st.markdown('<div class="card"><div class="card-title">📊 Emotion Distribution</div>', unsafe_allow_html=True)
        bars_html = ""
        for cls, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            e   = EMOJI.get(cls,'🎭')
            pct = p*100
            s   = EMOTION_STYLE.get(cls, {'bar':DEFAULT_BAR})
            bars_html += f"""
            <div class="prob-row">
                <div class="prob-emoji">{e}</div>
                <div class="prob-name">{cls}</div>
                <div class="prob-bg">
                    <div class="prob-fill" style="width:{pct:.1f}%;background:{s['bar']}"></div>
                </div>
                <div class="prob-pct">{pct:.1f}%</div>
            </div>"""
        st.markdown(bars_html + '</div>', unsafe_allow_html=True)
 
        # MODEL PERFORMANCE
        st.markdown("""
        <div class="card">
            <div class="card-title">🤖 Model Performance</div>
            <div class="perf-grid">
                <div class="perf-item">
                    <div class="perf-val">94.37%</div>
                    <div class="perf-lbl">Test Accuracy</div>
                </div>
                <div class="perf-item">
                    <div class="perf-val">Ensemble</div>
                    <div class="perf-lbl">Model Type</div>
                </div>
                <div class="perf-item">
                    <div class="perf-val">MFCC + Δ</div>
                    <div class="perf-lbl">Feature Type</div>
                </div>
                <div class="perf-item">
                    <div class="perf-val">4th / 14</div>
                    <div class="perf-lbl">Leaderboard</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    else:
        st.error('Could not analyze audio. Please try a different file.')
 
st.markdown('</div>', unsafe_allow_html=True)
 
st.markdown("""
<div class="footer">
    BIL216 Signals and Systems &nbsp;·&nbsp; Emo-Challenge 2026 Phase 3<br>
    <span style="color:#cbd5e1">Voting Ensemble: 2×SVM + LightGBM + XGBoost + MLP &nbsp;·&nbsp; 424 Features &nbsp;·&nbsp; Data Augmentation ×5</span>
</div>
""", unsafe_allow_html=True)
 
 
