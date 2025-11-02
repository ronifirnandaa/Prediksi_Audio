import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import tsfel
import joblib
import tempfile
import os
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder

# ==========================================================
# Halaman Streamlit
# ==========================================================
st.set_page_config(
    page_title="Klasifikasi Suara Buka/Tutup",
    page_icon="🎤",
    layout="wide"
)

# ==========================================================
# Load model dan artefak
# ==========================================================
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('artifacts/voice_classifier_model.pkl')
    scaler = joblib.load('artifacts/voice_scaler.pkl')
    feature_names = joblib.load('artifacts/feature_names.pkl')
    metadata = joblib.load('artifacts/model_metadata.pkl')
    cfg = tsfel.get_features_by_domain(["statistical"])
    return model, scaler, feature_names, metadata, cfg

# ==========================================================
# Preprocessing audio
# ==========================================================
def preprocess_audio(y, sr, target_sr=16000):
    if y is None or len(y) == 0:
        raise ValueError("File audio kosong.")
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) > 0 and np.max(np.abs(y_trimmed)) > 0:
        y_trimmed = y_trimmed / np.max(np.abs(y_trimmed))
    if len(y_trimmed) < sr * 0.1:
        raise ValueError("Audio terlalu pendek setelah trimming.")
    return y_trimmed, sr

# ==========================================================
# Ekstraksi fitur gabungan
# ==========================================================
def extract_combined_features(y, sr, feature_names, cfg):
    try:
        if y is None or len(y) < 2048:
            raise ValueError(f"Audio terlalu pendek (len={len(y)})")
        X_tsfel = tsfel.time_series_features_extractor(cfg, y, fs=sr, verbose=0)
        if X_tsfel.shape[0] == 0:
            raise ValueError("Hasil ekstraksi TSFEL kosong.")
        tsfel_feats = X_tsfel.iloc[0].values
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_feats = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        all_features = np.concatenate([tsfel_feats, mfcc_feats])
        if len(all_features) != len(feature_names):
            diff = len(feature_names) - len(all_features)
            all_features = np.append(all_features, np.zeros(diff)) if diff > 0 else all_features[:len(feature_names)]
        return all_features.reshape(1, -1)
    except Exception as e:
        st.error(f"Error ekstraksi fitur: {e}")
        return None

# ==========================================================
# Prediksi
# ==========================================================
def predict_audio(y, sr, model, scaler, feature_names, metadata, cfg):
    try:
        y_proc, sr_proc = preprocess_audio(y, sr, metadata["target_sr"])
        if len(y_proc) < 2048:
            st.warning("Audio terlalu pendek setelah preprocessing.")
            return None, None, None
        features = extract_combined_features(y_proc, sr_proc, feature_names, cfg)
        if features is None:
            return None, None, None
        features_df = pd.DataFrame(np.nan_to_num(features), columns=feature_names)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        return prediction, probabilities, y_proc
    except Exception as e:
        st.error(f"Error prediksi: {e}")
        return None, None, None

# ==========================================================
# Visualisasi audio
# ==========================================================
def plot_waveform_and_spectrogram(y, sr):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#2c3e50')
    axes[0].set_title("Waveform")
    axes[0].grid(alpha=0.3)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, hop_length=256, x_axis='time', y_axis='linear', ax=axes[1], cmap='plasma')
    axes[1].set_title("Spectrogram")
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    plt.tight_layout()
    return fig

# ==========================================================
# Main App
# ==========================================================
def main():
    st.markdown("<h1 style='text-align:center;color:#2c3e50;'>Klasifikasi Suara Buka/Tutup</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#34495e;'>Prediksi berbasis TSFEL + MFCC + Random Forest</p>", unsafe_allow_html=True)

    with st.spinner("Memuat model..."):
        model, scaler, feature_names, metadata, cfg = load_model_artifacts()

    # ========================== Sidebar ==========================
    st.sidebar.title("Informasi Model")
    st.sidebar.markdown(f"""
    **Akurasi Training:** {metadata['train_accuracy']:.2%}  
    **Akurasi Testing:** {metadata['test_accuracy']:.2%}  
    **Jumlah Fitur:** {metadata['n_features']}  
    **Sample Rate Target:** {metadata['target_sr']} Hz
    """)
    st.sidebar.markdown("---")
    st.sidebar.title("Petunjuk Penggunaan")
    st.sidebar.markdown("""
    1. Pilih metode input audio (rekam langsung atau upload).  
    2. Klik tombol Prediksi untuk memproses audio.  
    3. Lihat hasil prediksi dan visualisasi.
    """)

    # ========================== Input Audio ==========================
    st.subheader("Input Audio")
    method = st.radio("Metode input:", ["Rekam Langsung", "Upload File"])
    audio_data, sr = None, None

    if method == "Rekam Langsung":
        audio_bytes = audio_recorder(text="Mulai / Stop Rekam", recording_color="#e74c3c", neutral_color="#3498db")
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
                st.audio(audio_bytes, format="audio/wav")
                audio_data = y
            finally:
                os.remove(tmp_path)
    else:
        uploaded_file = st.file_uploader("Upload file audio (.wav / .mp3)", type=["wav", "mp3"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
                st.audio(uploaded_file, format="audio/wav")
                audio_data = y
            finally:
                os.remove(tmp_path)

    # ========================== Prediksi ==========================
    if audio_data is not None and sr is not None:
        durasi = len(audio_data) / sr
        st.info(f"Durasi audio: {durasi:.2f} detik | Sample rate: {sr} Hz")

        if len(audio_data) < 2048:
            st.warning("Audio terlalu pendek. Rekam ulang minimal 0.5 detik.")
        elif st.button("Prediksi"):
            with st.spinner("Memproses audio..."):
                pred, prob, y_proc = predict_audio(audio_data, sr, model, scaler, feature_names, metadata, cfg)
                if pred is not None:
                    label = metadata["label_map"][pred]
                    confidence = prob[pred] * 100

                    # ===== Card Label Utama =====
                    st.markdown(
                        f"<div style='border-radius:12px; padding:20px; background:#f0f3f5; text-align:center;'>"
                        f"<h1 style='color:#2c3e50;'>{label}</h1>"
                        f"<h3 style='color:#34495e;'>Confidence: {confidence:.2f}%</h3>"
                        f"</div>", unsafe_allow_html=True
                    )

                    # ===== Probabilitas Kelas (Minimalis) =====
                    st.subheader("Probabilitas Kelas")
                    for i, name in metadata["label_map"].items():
                        st.markdown(f"**{name}**: {prob[i]*100:.2f}%")
                        st.progress(prob[i])

                    # ===== Visualisasi Audio =====
                    st.subheader("Visualisasi Audio")
                    fig = plot_waveform_and_spectrogram(y_proc, metadata["target_sr"])
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
