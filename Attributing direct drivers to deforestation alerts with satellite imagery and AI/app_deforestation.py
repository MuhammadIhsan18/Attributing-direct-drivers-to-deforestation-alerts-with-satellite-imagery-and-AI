import streamlit as st
import pandas as pd
import plotly.express as px

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deforestation Driver AI Monitor", page_icon="🌲", layout="wide")

st.title("🌲 Dasbor AI: Pemantauan Pemicu Deforestasi & Satelit")
st.caption("Studi Kasus: Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI (Dr. Bart Slagter - Wageningen University)")
st.markdown("---")

# --- 1. DATA DUMMY DRIVER DEFORESTASI ---
data_forest = {
    "Region": ["Amazon (Brazil)", "Amazon (Brazil)", "Congo Basin", "Congo Basin", "Borneo (Indonesia)", "Borneo (Indonesia)", "Bolivia", "Bolivia"],
    "Driver_Penyebab": ["Pertanian Skala Besar", "Kebakaran Hutan", "Pertanian Skala Kecil", "Pembalakan Selektif", "Pertanian Skala Besar", "Pertambangan", "Kebakaran Hutan", "Pembalakan Selektif"],
    "Luas_Deforestasi_Hectares": [14500, 8200, 3100, 4500, 6700, 2900, 5400, 3800],
    "Confidence_Model": ["Tinggi", "Sedang", "Tinggi", "Tinggi", "Tinggi", "Sedang", "Rendah", "Tinggi"],
    "Tahun": [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024]
}
df_def = pd.DataFrame(data_forest)

# --- 2. SIDEBAR FILTER ---
st.sidebar.header("🔍 Filter Wilayah Hutan")
pilih_region = st.sidebar.multiselect(
    "Pilih Wilayah (Region):",
    options=df_def["Region"].unique(),
    default=df_def["Region"].unique()
)

df_filtered = df_def[df_def["Region"].isin(pilih_region)]

# --- 3. METRIK UTAMA ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Luas Hilangnya Hutan", f"{df_filtered['Luas_Deforestasi_Hectares'].sum():,} Ha", "Berdasarkan Alert Satelit")
col2.metric("Sensor Utama Digunakan", "Sentinel-1 (Radar) & Sentinel-2", "Resolusi 10m")
col3.metric("Akurasi Rata-rata Model AI", "84.5%", "Validasi Silang")

st.markdown("---")

# --- 4. VISUALISASI PLOTLY ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Distribusi Pemicu Utama (Direct Drivers)")
    fig_bar = px.bar(
        df_filtered,
        x="Driver_Penyebab",
        y="Luas_Deforestasi_Hectares",
        color="Region",
        title="Luas Kerusakan Hutan Berdasarkan Pemicu AI",
        text_auto=True
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.subheader("🌍 Proporsi Driver di Berbagai Wilayah")
    fig_pie = px.pie(
        df_filtered,
        values="Luas_Deforestasi_Hectares",
        names="Driver_Penyebab",
        title="Persentase Penyebab Deforestasi",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# --- 5. TABEL DATA & CATATAN ---
st.subheader("📋 Log Klasifikasi AI Berbasis Citra Satelit")
st.dataframe(df_filtered, use_container_width=True, hide_index=True)

st.info("💡 **Catatan Riset WUR:** Kombinasi citra optik (Sentinel-2) dan radar (Sentinel-1) yang diproses melalui *Convolutional Neural Network* (CNN) memungkinkan deteksi dini perubahan hutan tiap bulan secara transparan.")
