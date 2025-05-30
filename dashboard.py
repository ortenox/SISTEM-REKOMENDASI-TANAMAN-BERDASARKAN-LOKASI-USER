import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set halaman
st.set_page_config(page_title="Deteksi Musim", layout="centered")
st.title("📅 Deteksi Musim Otomatis")

# Load Data
df = pd.read_csv("data.csv")
df['tanggal'] = pd.date_range(start='2025-06-01', periods=len(df), freq='D')

# Pilih Mode
mode = st.radio("Pilih Mode:", ["Per Tanggal", "Rentang Waktu"])

# Mode: Per Tanggal
if mode == "Per Tanggal":
    st.subheader("Pilih Tanggal")
    selected_date = st.date_input("Tanggal", min_value=df['tanggal'].min(), max_value=df['tanggal'].max(), value=df['tanggal'].min())
    row = df[df['tanggal'] == pd.to_datetime(selected_date)]
    if not row.empty:
        rainfall = row['rainfall'].values[0]
        musim = "Musim Hujan" if rainfall > 150 else "Musim Kemarau"
        st.markdown("---")
        st.write(f"📅 **Tanggal:** {selected_date.strftime('%d %B %Y')}")
        st.metric("🌧️ Curah Hujan (mm)", f"{rainfall:.2f}")
        st.success(f"🟢 **Musim:** {musim}")
    else:
        st.warning("Data tidak ditemukan untuk tanggal tersebut.")

# Mode: Rentang Waktu
else:
    st.subheader("Pilih Rentang Tanggal")
    col1, col2 = st.columns(2)

    min_date = df['tanggal'].min().date()
    max_date = df['tanggal'].max().date()

    with col1:
        start_date = st.date_input(
            "Dari",
            min_value=min_date,
            max_value=max_date,
            value=min_date
        )

    with col2:
        # Hitung default end_date
        default_end = start_date + pd.Timedelta(days=6)
        if default_end > max_date:
            default_end = max_date

        end_date = st.date_input(
            "Sampai",
            min_value=start_date,
            max_value=max_date,
            value=default_end
        )
    if start_date > end_date:
        st.error("Tanggal akhir harus setelah tanggal awal.")
    else:
        mask = (df['tanggal'] >= pd.to_datetime(start_date)) & (df['tanggal'] <= pd.to_datetime(end_date))
        filtered = df[mask]

        if not filtered.empty:
            avg_rainfall = filtered['rainfall'].mean()
            musim_dominan = "Musim Hujan" if avg_rainfall > 150 else "Musim Kemarau"
            st.markdown("---")
            st.write(f"📆 **Periode:** {start_date.strftime('%d %B %Y')} - {end_date.strftime('%d %B %Y')}")
            st.metric("📊 Rata-rata Curah Hujan (mm)", f"{avg_rainfall:.2f}")
            st.success(f"🟢 **Musim Dominan:** {musim_dominan}")

            # Visualisasi Tren Rainfall
            st.markdown("### Tren Curah Hujan Harian")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(filtered['tanggal'], filtered['rainfall'], marker='o', linestyle='-')
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Curah Hujan (mm)")
            ax.set_title("Tren Curah Hujan")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("Tidak ada data dalam rentang waktu tersebut.")