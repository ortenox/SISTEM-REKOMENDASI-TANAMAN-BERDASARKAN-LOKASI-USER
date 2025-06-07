import joblib
import numpy as np
import pandas as pd

# Load model dan preprocessor
model = joblib.load("crop_recommendation_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def predict_crop(N, temperature, humidity, ph, rainfall):
    """
    Melakukan prediksi tanaman berdasarkan input cuaca dan tanah.
    Args:
        N (float): Nitrogen
        temperature (float): Suhu rata-rata
        humidity (float): Kelembaban rata-rata
        ph (float): Keasaman tanah
        rainfall (float): Curah hujan rata-rata
    Returns:
        tuple: (nama tanaman yang direkomendasikan, persentase keyakinan)
    """
    # Buat dataframe input
    input_df = pd.DataFrame([[N, temperature, humidity, ph, rainfall]],
                            columns=["N", "temperature", "humidity", "ph", "rainfall"])
    
    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediksi
    pred = model.predict(input_scaled)
    predicted_label = np.argmax(pred)
    crop_name = label_encoder.inverse_transform([predicted_label])[0]
    confidence = np.max(pred) * 100

    return crop_name, confidence
