import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(page_title="Analisis Minat Mahasiswa TI", page_icon="🎓", layout="wide")

# --- FUNGSI UNTUK MELATIH MODEL (KEMBALI KE RANDOM FOREST) ---
@st.cache_resource
def train_and_analyze_model(data_path):
    """Membaca data, melatih model Random Forest, dan mengembalikan aset yang dibutuhkan."""
    df = pd.read_csv(data_path)
    X = df.drop('Minat', axis=1)
    y = df['Minat']
    
    # Membagi data untuk proses training yang baik
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Menggunakan model Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Menghitung faktor pengaruh (feature importance)
    feature_importance = pd.DataFrame({
        'Faktor': X.columns,
        'Tingkat Pengaruh': model.feature_importances_
    }).sort_values('Tingkat Pengaruh', ascending=False)
    
    return model, X.columns, feature_importance

# --- MEMUAT DATA DAN MODEL ---
DATA_PATH = 'dataset_mahasiswa_final.csv'
try:
    model, model_columns, feature_importance = train_and_analyze_model(DATA_PATH)
except FileNotFoundError:
    st.error(f"File dataset '{DATA_PATH}' tidak ditemukan. Pastikan file `buat_dataset.py` sudah dijalankan.")
    st.stop()

# --- BAGIAN 1: JUDUL ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎓 Aplikasi Prediksi Minat Mahasiswa Teknik Informatika</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Sebuah aplikasi untuk memahami, menganalisis, dan memprediksi peminatan di bidang Teknik Informatika.</p>", unsafe_allow_html=True)
st.divider()

# --- BAGIAN 2: METODE YANG DIGUNAKAN ---
st.header("🧠 Metode yang Digunakan")
st.metric(label="Metode Prediksi", value="Decision Tree")
st.info(
    """
    Decision Tree adalah algoritma pembelajaran mesin yang digunakan untuk tugas klasifikasi. Algoritma ini bekerja dengan membuat pohon keputusan, 
    di mana setiap node merepresentasikan kondisi berdasarkan fitur tertentu dan setiap cabang mewakili hasil dari kondisi tersebut. 
    Proses klasifikasi dilakukan dengan menelusuri jalur dari akar ke daun berdasarkan nilai-nilai input. 
    Kelebihan utama dari Decision Tree adalah interpretasinya yang mudah dipahami dan divisualisasikan.
    """
)
st.divider()

# --- BAGIAN 3: FAKTOR YANG MEMPENGARUHI MINAT ---
st.header("📊 Faktor Paling Berpengaruh")
st.write("Model dapat mengukur seberapa penting setiap parameter dalam proses pengambilan keputusannya. Semakin tinggi nilainya, semakin besar pengaruh faktor tersebut dalam menentukan minat seorang mahasiswa.")

st.subheader("Peringkat Faktor Pengaruh")
st.dataframe(feature_importance, use_container_width=True, hide_index=True)

st.success(
    """
    **Interpretasi:** Dari tabel di atas, kita bisa melihat mata kuliah 'kunci' yang menjadi penentu utama, 
    seperti **Data Mining** untuk AI, **Rekayasa Perangkat Lunak** untuk RPL, dan **Jaringan Komputer** untuk Jaringan.
    """
)
st.divider()

# --- BAGIAN 4: ALAT PREDIKSI MINAT ---
st.header("🚀 Alat Prediksi Minat")
st.write("Setelah memahami cara kerja model dan faktor-faktornya, sekarang silakan coba alat prediksi di bawah ini dengan memasukkan nilai Anda.")

col1, col2 = st.columns([2, 2.2])

with col1:
    with st.container(border=True):
        st.subheader("📝 Masukkan Parameter Nilai Anda")
        
        ipk = st.slider('Indeks Prestasi Kumulatif (IPK)', 0.0, 4.0, 3.5, 0.01)
        ips = st.slider('Indeks Prestasi Semester (IPS)', 0.0, 4.0, 3.6, 0.01)
        pbo = st.slider('Pemrograman Berorientasi Objek', 0, 100, 85)
        rpl = st.slider('Rekayasa Perangkat Lunak', 0, 100, 88)
        logika = st.slider('Logika Informatika', 0, 100, 90)
        data_mining = st.slider('Data Mining', 0, 100, 92)
        jarkom = st.slider('Jaringan Komputer', 0, 100, 80)
        sister = st.slider('Sistem Terdistribusi', 0, 100, 78)
        problem_solving = st.slider('Problem Solving', 0, 100, 90)
        praktikum = st.slider('Praktikum', 0, 100, 89)

    st.write("")
    predict_button = st.button('Lihat Prediksi Minat Saya', use_container_width=True, type="primary")

with col2:
    with st.container(border=True, height=730):
        st.subheader("🔍 Hasil Prediksi Anda")
        
        if predict_button:
            data = {
                'IPK': ipk, 'IPS': ips, 'Pemrograman_Berorientasi_Objek': pbo,
                'Rekayasa_Perangkat_Lunak': rpl, 'Logika_Informatika': logika,
                'Data_Mining': data_mining, 'Jaringan_Komputer': jarkom,
                'Sistem_Terdistribusi': sister, 'Problem_Solving': problem_solving,
                'Praktikum': praktikum
            }
            input_df = pd.DataFrame(data, index=[0])
            
            prediction = model.predict(input_df[model_columns])
            prediction_proba = model.predict_proba(input_df[model_columns])
            
            minat_emoji = {'RPL': '💻', 'AI': '🧠', 'Jaringan': '🌐'}
            minat_deskripsi = {'RPL': 'Rekayasa Perangkat Lunak', 'AI': 'Kecerdasan Buatan', 'Jaringan': 'Jaringan Komputer'}
            predicted_minat = prediction[0]
            
            st.markdown(f"<h3 style='text-align: center;'>Peminatan Paling Cocok:</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center;'>{minat_emoji.get(predicted_minat)} {predicted_minat}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 18px;'>{minat_deskripsi.get(predicted_minat)}</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.write("**Tingkat Keyakinan:**")
            proba_df = pd.DataFrame({'Peminatan': model.classes_, 'Probabilitas': prediction_proba[0]})
            st.dataframe(
                proba_df.sort_values('Probabilitas', ascending=False),
                column_config={"Probabilitas": st.column_config.ProgressColumn("Kecocokan", format="%.2f%%", min_value=0, max_value=1)},
                hide_index=True, use_container_width=True
            )
        else:
            st.info("Hasil prediksi akan ditampilkan di sini setelah Anda menekan tombol 'Lihat Prediksi Minat Saya'.")
