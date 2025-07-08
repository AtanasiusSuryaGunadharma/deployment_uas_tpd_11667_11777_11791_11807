# app.py
import streamlit as st
import pandas as pd
import pickle
import os

# Fungsi untuk memuat model dan komponen lainnya
def load_model_data(path):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data
    else:
        st.error(f"File model '{path}' tidak ditemukan. Harap jalankan script `create_model_final.py` terlebih dahulu.")
        return None

# Memuat data
model_data = load_model_data(r'student_performance_model.pkl')

# CSS Kustom untuk tema
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: #ffffff; }
    .stButton button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 24px; font-size: 18px; }
    h1, h2 { font-family: 'Courier New', monospace; color: #4CAF50; text-align: center; }
    p, label { font-family: 'Arial', sans-serif; }
    .st-emotion-cache-1kyxreq { border: 1px solid #4CAF50; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- Antarmuka Aplikasi ---
st.markdown("<h1>🎓 Prediksi Klaster Performa Siswa</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Aplikasi ini memprediksi klaster performa siswa (Rendah, Sedang, Tinggi) menggunakan model terbaik yaitu Random Forest.</p>", unsafe_allow_html=True)
st.markdown("---")

# Hanya tampilkan antarmuka jika model berhasil dimuat
if model_data:
    model = model_data["model"]
    encoders = model_data["encoders"]
    cluster_labels = model_data["cluster_labels"]
    feature_names = model_data["feature_names"]

    st.sidebar.header("📝 Masukkan Data Siswa")

    # Membuat input form di sidebar menggunakan opsi dari encoder yang disimpan
    inputs = {}
    inputs['jenis_kelamin'] = st.sidebar.selectbox("Jenis Kelamin", encoders['jenis_kelamin'].classes_)
    inputs['ras_etnis'] = st.sidebar.selectbox("Ras/Etnis", encoders['ras_etnis'].classes_)
    inputs['pendidikan_orangtua'] = st.sidebar.selectbox("Pendidikan Orang Tua", encoders['pendidikan_orangtua'].classes_)
    inputs['makan_siang'] = st.sidebar.selectbox("Tipe Makan Siang", encoders['makan_siang'].classes_)
    inputs['kursus_persiapan'] = st.sidebar.selectbox("Kursus Persiapan", encoders['kursus_persiapan'].classes_)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Nilai IP (0.00 - 4.00)")

    inputs['ip_matematika'] = st.sidebar.number_input("IP Matematika", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    inputs['ip_membaca']   = st.sidebar.number_input("IP Membaca",   min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    inputs['ip_menulis']   = st.sidebar.number_input("IP Menulis",   min_value=0.0, max_value=4.0, value=3.0, step=0.01)


    # Tombol prediksi
    if st.sidebar.button("🚀 Prediksi Performa"):
        # Buat DataFrame dari input
        input_df = pd.DataFrame([inputs])

        # Proses encoding untuk setiap kolom kategorikal
        for col, encoder in encoders.items():
            input_df[col] = encoder.transform(input_df[col])

        # Pastikan urutan kolom sama persis dengan saat training
        input_df = input_df[feature_names]

        # Lakukan prediksi
        prediction = model.predict(input_df)
        predicted_cluster_id = prediction[0]
        predicted_cluster_label = cluster_labels[predicted_cluster_id]

        # Tampilkan hasil
        st.markdown(f"<h2>Hasil Prediksi</h2>", unsafe_allow_html=True)
        
        result_style = """
            border: 2px solid {color}; 
            border-radius: 10px; 
            padding: 20px; 
            text-align: center; 
            font-size: 24px;
        """
        
        if predicted_cluster_label == 'Tinggi':
            st.markdown(f"<div style='{result_style.format(color='#28a745')}'>Klaster: <strong>Tinggi</strong> 🏆</div>", unsafe_allow_html=True)
            st.success("Siswa ini menunjukkan potensi akademik yang sangat baik.")
        elif predicted_cluster_label == 'Sedang':
            st.markdown(f"<div style='{result_style.format(color='#ffc107')}'>Klaster: <strong>Sedang</strong> 👍</div>", unsafe_allow_html=True)
            st.info("Siswa ini memiliki performa akademik yang cukup baik.")
        else:
            st.markdown(f"<div style='{result_style.format(color='#dc3545')}'>Klaster: <strong>Rendah</strong> 📚</div>", unsafe_allow_html=True)
            st.warning("Siswa ini mungkin memerlukan perhatian atau bimbingan tambahan.")
            
        st.markdown("---")
        with st.expander("Lihat Detail Input yang Diproses"):
            st.write("Data mentah dari input:")
            st.json(inputs)
            st.write("Data setelah di-encode (yang digunakan model):")
            st.dataframe(input_df)
