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
    page_icon="🎙️",
    layout="centered"
)

# ── Duygu emojileri ───────────────────────────────────────────
EMOJI = {
    'Furious': '😡',
    'Happy':   '😊',
    'Neutral': '😐',
    'Sad':     '😢',
    'Shocked': '😲'
}

RENK = {
    'Furious': '#e74c3c',
    'Happy':   '#f1c40f',
    'Neutral': '#95a5a6',
    'Sad':     '#3498db',
    'Shocked': '#9b59b6'
}

# ── Model yükleme ─────────────────────────────────────────────
MODEL_DIR = 'models'

@st.cache_resource
def load_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Google Drive dosya ID'leri
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

# ── Tahmin fonksiyonu ─────────────────────────────────────────
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
st.title("🎙️ Emo-Challenge 2026")
st.markdown("### Gerçek Zamanlı Duygu Analizi")
st.markdown("**Group 12** | Ayşegül Muhtaç · Sena Poyraz · Yiğit Kadir Gökdemir")
st.markdown("---")

# Model yükle
with st.spinner('🤖 Model yükleniyor...'):
    try:
        model, scaler, selector, le = load_models()
        st.success('✅ Model hazır! (%94.37 doğruluk)')
    except Exception as e:
        st.error(f'❌ Model yüklenemedi: {e}')
        st.stop()

st.markdown("---")

# Dosya yükleme
st.markdown("### 📂 Ses Dosyası Yükle")
uploaded = st.file_uploader(
    "Bir WAV veya MP3 dosyası seçin",
    type=['wav', 'mp3'],
    help="Maksimum 3 saniyelik ses önerilir"
)

if uploaded:
    st.audio(uploaded, format='audio/wav')

    with st.spinner('🔍 Analiz ediliyor...'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        duygu, olasiliklar = tahmin_yap(tmp_path, model, scaler, selector, le)
        os.unlink(tmp_path)

    if duygu:
        emoji = EMOJI.get(duygu, '🎭')
        renk = RENK.get(duygu, '#333')

        st.markdown("---")
        st.markdown(f"""
        <div style='text-align:center; padding:30px;
                    background-color:{renk}22;
                    border-radius:15px;
                    border: 2px solid {renk}'>
            <h1 style='color:{renk}; font-size:80px; margin:0'>{emoji}</h1>
            <h2 style='color:{renk}'>{duygu}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📊 Tüm Sınıf Olasılıkları")
        for sinif, olasilik in sorted(olasiliklar.items(),
                                       key=lambda x: x[1], reverse=True):
            e = EMOJI.get(sinif, '🎭')
            st.markdown(f"{e} **{sinif}**")
            st.progress(float(olasilik),
                        text=f"%{olasilik*100:.1f}")
    else:
        st.error('❌ Ses analiz edilemedi, lütfen başka bir dosya deneyin.')

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:12px'>
BIL216 Signals and Systems | Emo-Challenge 2026 Faz 3<br>
Model: Voting Ensemble (2×SVM + LightGBM + XGBoost + MLP) | Doğruluk: %94.37
</div>
""", unsafe_allow_html=True)
