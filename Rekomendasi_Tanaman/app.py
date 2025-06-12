import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from opencage.geocoder import OpenCageGeocode
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib

# Konfigurasi halaman dan styling
st.set_page_config(page_title="Dashboard Cuaca & Tanah", layout="centered")

st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Rekomendasi Tanaman Berdasarkan Cuaca & Tanah")

# Load model AI
model = load_model("Rekomendasi_Tanaman/crop_recommendation_model.h5")
scaler = joblib.load("Rekomendasi_Tanaman/scaler.pkl")
le = joblib.load("Rekomendasi_Tanaman/label_encoder.pkl")

API_KEY = "d4179fb703532ad460882dd59234d867"
OPENCAGE_KEY = "bb5f4c87ead0454dac0231fd0b10dd19"

def get_coords_from_city(city_name):
    try:
        geocoder = OpenCageGeocode(OPENCAGE_KEY)
        results = geocoder.geocode(city_name)
        if results and len(results):
            return results[0]['geometry']['lat'], results[0]['geometry']['lng']
    except Exception as e:
        st.error(f"Geocoding error: {e}")
    return None, None

def get_forecast_summary(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    temps, humidities, rainfalls = [], [], []
    for entry in data["list"]:
        temps.append(entry['main']['temp'])
        humidities.append(entry['main']['humidity'])
        rain = entry.get('rain', {}).get('3h', 0)
        rainfalls.append(rain)
    return {
        "temperature": round(sum(temps) / len(temps), 2),
        "humidity": round(sum(humidities) / len(humidities), 2),
        "rainfall": round(sum(rainfalls) / len(rainfalls), 2)
    }

def get_soil_data(lat, lon):
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    def get_value(prop):
        params = {
            "property": prop,
            "depth": "0-5cm",
            "value": "mean",
            "lat": lat,
            "lon": lon
        }
        r = requests.get(base_url, params=params)
        if r.status_code == 200:
            try:
                val = r.json()["properties"]["layers"][0]["depths"][0]["values"]["mean"]
                return val / 10.0 if prop == "phh2o" else val
            except:
                return None
        return None
    return {
        "ph": get_value("phh2o"),
        "N": get_value("nitrogen")
    }

# Input kota
city = st.text_input("Masukkan nama kota (Contoh: Bandung)")

lat, lon = None, None
if city:
    lat, lon = get_coords_from_city(city)
    if lat and lon:
        st.success(f"Koordinat: ({lat:.2f}, {lon:.2f})")
    else:
        st.warning("Gagal mendapatkan koordinat dari nama kota.")

# Peta interaktif
st.subheader("Klik lokasi di peta (opsional)")
m = folium.Map(location=[-2.5, 118], zoom_start=5)
folium.LatLngPopup().add_to(m)
map_data = st_folium(m, width=700, height=500)

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"Koordinat dipilih: ({lat:.2f}, {lon:.2f})")

if lat and lon:
    with st.container():
        st.subheader("Ringkasan Cuaca")
        cuaca = get_forecast_summary(lat, lon, API_KEY)
        if cuaca:
            for k, v in cuaca.items():
                st.write(f"**{k.replace('_', ' ').title()}:** {v}")
        else:
            st.error("Gagal mendapatkan data cuaca.")

        st.subheader("Data Tanah")
        tanah = get_soil_data(lat, lon)

        default_N = 54.2
        default_ph = 6.4

        N_val = tanah.get("N")
        ph_val = tanah.get("ph")

        if N_val is None:
            st.write(f"N: {default_N}")
            N_val = default_N
        else:
            st.write(f"N: {N_val}")

        if ph_val is None:
            st.write(f"pH: {default_ph}")
            ph_val = default_ph
        else:
            st.write(f"pH: {ph_val}")

    with st.container():
        if cuaca and all(val is not None for val in [cuaca.get("temperature"), cuaca.get("humidity"), cuaca.get("rainfall"), N_val, ph_val]):
            st.subheader("Rekomendasi AI")
            input_df = pd.DataFrame([[N_val, cuaca["temperature"], cuaca["humidity"], ph_val, cuaca["rainfall"]]],
                                    columns=["N", "temperature", "humidity", "ph", "rainfall"])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)
            pred_crop = le.inverse_transform([np.argmax(pred)])
            confidence = np.max(pred) * 100

            st.success(f"Tanaman yang direkomendasikan: **{pred_crop[0]}**")
            st.write(f"Keyakinan model: **{confidence:.2f}%**")
        else:
            st.warning("Data tidak lengkap untuk melakukan prediksi AI.")
