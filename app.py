import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
import os
import gdown
 
# ── Sayfa ayarları ────────────────────────────────────────────
st.set_page_config(
    page_title="Emo-Challenge 2026",
    page_icon="🎭",
    layout="centered"
)
 
# ── Özel CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
 
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }
 
    .hero-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 40px 30px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
    }
 
    .hero-title {
        font-size: 48px;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -1px;
    }
 
    .hero-sub {
        font-size: 18px;
        color: rgba(255,255,255,0.85);
        margin-top: 8px;
    }
 
    .hero-team {
        font-size: 14px;
        color: rgba(255,255,255,0.65);
        margin-top: 12px;
        letter-spacing: 1px;
    }
 
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 50px;
        padding: 6px 18px;
        font-size: 13px;
        color: white;
        margin-top: 14px;
        backdrop-filter: blur(10px);
    }
 
    .upload-box {
        background: rgba(255,255,255,0.05);
        border: 2px dashed rgba(102,126,234,0.6);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s;
    }
 
    .result-card {
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        margin: 25px 0;
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        animation: fadeIn 0.5s ease;
    }
 
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
 
    .result-emoji {
        font-size: 90px;
        margin-bottom: 10px;
        display: block;
    }
 
    .result-label {
        font-size: 36px;
        font-weight: 800;
        margin: 0;
    }
 
    .prob-label {
        color: rgba(255,255,255,0.9);
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 4px;
    }
 
    .section-title {
        color: rgba(255,255,255,0.9);
        font-size: 20px;
        font-weight: 700;
        margin: 25px 0 15px 0;
        padding-left: 5px;
        border-left: 4px solid #667eea;
    }
 
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.35);
        font-size: 12px;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
 
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }
 
    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)
 
# ── Duygu ayarları ────────────────────────────────────────────
EMOJI = {
    'Furious': '😡',
    'Happy':   '😊',
    'Neutral': '😐',
    'Sad':     '😢',
    'Shocked': '😲'
}
 
GRADIENT = {
    'Furious': 'linear-gradient(135deg, #ff416c, #ff4b2b)',
    'Happy':   'linear-gradient(135deg, #f7971e, #ffd200)',
    'Neutral': 'linear-gradient(135deg, #4facfe, #00f2fe)',
    'Sad':     'linear-gradient(135deg, #4776e6, #8e54e9)',
    'Shocked': 'linear-gradient(135deg, #a18cd1, #fbc2eb)'
}
 
RENK = {
    'Furious': '#ff416c',
    'Happy':   '#f7971e',
    'Neutral': '#4facfe',
    'Sad':     '#4776e6',
    'Shocked': '#a18cd1'
}
 
# ── Model yükleme ─────────────────────────────────────────────
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
    for filename, file_id in files.items():
        path = f'{MODEL_DIR}/{filename}'
        if not os.path.exists(path):
            with st.spinner(f'📥 {filename} yükleniyor...'):
                gdown.download(
                    f'https://drive.google.com/uc?id={file_id}',
                    path, quiet=True
                )
    with open(f'{MODEL_DIR}/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{MODEL_DIR}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{MODEL_DIR}/selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    with open(f'{MODEL_DIR}/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, selector, le
 
# ── Öznitelik çıkarma ─────────────────────────────────────────
def ozellik_cikar(y, sr):
    try:
        ozellikler = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
        ozellikler.extend(np.mean(mfcc, axis=1))
        ozellikler.extend(np.std(mfcc, axis=1))
        mfcc_delta = librosa.feature.delta(mfcc)
        ozellikler.extend(np.mean(mfcc_delta, axis=1))
        ozellikler.extend(np.std(mfcc_delta, axis=1))
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        ozellikler.extend(np.mean(mfcc_delta2, axis=1))
        ozellikler.extend(np.std(mfcc_delta2, axis=1))
        zcr = librosa.feature.zero_crossing_rate(y)
        ozellikler.append(np.mean(zcr))
        ozellikler.append(np.std(zcr))
        rms = librosa.feature.rms(y=y)
        ozellikler.append(np.mean(rms))
        ozellikler.append(np.std(rms))
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        ozellikler.append(np.mean(spec_centroid))
        ozellikler.append(np.std(spec_centroid))
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        ozellikler.append(np.mean(spec_bw))
        ozellikler.append(np.std(spec_bw))
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        ozellikler.append(np.mean(rolloff))
        ozellikler.append(np.std(rolloff))
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        ozellikler.extend(np.mean(spec_contrast, axis=1))
        ozellikler.extend(np.std(spec_contrast, axis=1))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        ozellikler.extend(np.mean(chroma, axis=1))
        ozellikler.extend(np.std(chroma, axis=1))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        ozellikler.append(np.mean(mel_db))
        ozellikler.append(np.std(mel_db))
        ozellikler.append(np.max(mel_db))
        ozellikler.append(np.min(mel_db))
        try:
            y_harm = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
            ozellikler.extend(np.mean(tonnetz, axis=1))
            ozellikler.extend(np.std(tonnetz, axis=1))
        except:
            ozellikler.extend([0]*12)
        return np.array(ozellikler, dtype=np.float32)
    except:
        return None
 
def tahmin_yap(audio_path, model, scaler, selector, le):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    feat = ozellik_cikar(y, sr)
    if feat is None:
        return None, None
    feat_scaled = scaler.transform([feat])
    feat_sel = selector.transform(feat_scaled)
    pred = model.predict(feat_sel)[0]
    proba = model.predict_proba(feat_sel)[0]
    duygu = le.inverse_transform([pred])[0]
    return duygu, dict(zip(le.classes_, proba))
 
# ── ARAYÜZ ───────────────────────────────────────────────────
 
# Hero
st.markdown("""
<div class="hero-box">
    <div class="result-emoji">🎭</div>
    <p class="hero-title">Emo-Challenge 2026</p>
    <p class="hero-sub">Gerçek Zamanlı Duygu Analizi</p>
    <p class="hero-team">GROUP 12 &nbsp;·&nbsp; Ayşegül Muhtaç &nbsp;·&nbsp; Sena Poyraz &nbsp;·&nbsp; Yiğit Kadir Gökdemir</p>
    <span class="badge">🤖 Voting Ensemble &nbsp;|&nbsp; %94.37 Doğruluk</span>
</div>
""", unsafe_allow_html=True)
 
# Model yükle
with st.spinner('🤖 Model yükleniyor, lütfen bekleyin...'):
    try:
        model, scaler, selector, le = load_models()
        st.success('✅ Model hazır!')
    except Exception as e:
        st.error(f'❌ Model yüklenemedi: {e}')
        st.stop()
 
# Upload
st.markdown('<p class="section-title">🎙️ Ses Dosyası Yükle</p>', unsafe_allow_html=True)
st.markdown("WAV veya MP3 formatında bir ses dosyası yükleyin. Model sesin duygusunu analiz edecek.")
 
uploaded = st.file_uploader(
    "",
    type=['wav', 'mp3'],
    help="Maksimum 200MB, WAV veya MP3 formatı"
)
 
if uploaded:
    st.audio(uploaded, format='audio/wav')
    st.markdown('<p class="section-title">🔍 Analiz Sonucu</p>', unsafe_allow_html=True)
 
    with st.spinner('⚡ Ses analiz ediliyor...'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        duygu, olasiliklar = tahmin_yap(tmp_path, model, scaler, selector, le)
        os.unlink(tmp_path)
 
    if duygu:
        emoji = EMOJI.get(duygu, '🎭')
        gradient = GRADIENT.get(duygu, 'linear-gradient(135deg, #667eea, #764ba2)')
        renk = RENK.get(duygu, '#667eea')
 
        st.markdown(f"""
        <div class="result-card" style="background: {gradient};">
            <span class="result-emoji">{emoji}</span>
            <p class="result-label" style="color:white">{duygu}</p>
            <p style="color:rgba(255,255,255,0.8); font-size:14px; margin-top:8px">
                Güven: %{olasiliklar[duygu]*100:.1f}
            </p>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown('<p class="section-title">📊 Tüm Duygu Olasılıkları</p>', unsafe_allow_html=True)
 
        for sinif, olasilik in sorted(olasiliklar.items(), key=lambda x: x[1], reverse=True):
            e = EMOJI.get(sinif, '🎭')
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f'<p class="prob-label">{e} {sinif}</p>', unsafe_allow_html=True)
                st.progress(float(olasilik))
            with col2:
                st.markdown(f'<p style="color:white; font-weight:700; padding-top:25px">%{olasilik*100:.1f}</p>', unsafe_allow_html=True)
    else:
        st.error('❌ Ses analiz edilemedi, lütfen başka bir dosya deneyin.')
 
# Footer
st.markdown("""
<div class="footer">
    BIL216 Signals and Systems &nbsp;|&nbsp; Emo-Challenge 2026 Faz 3<br>
    Voting Ensemble: 2×SVM + LightGBM + XGBoost + MLP &nbsp;|&nbsp; 424 Öznitelik &nbsp;|&nbsp; Data Augmentation
</div>
""", unsafe_allow_html=True)
 
