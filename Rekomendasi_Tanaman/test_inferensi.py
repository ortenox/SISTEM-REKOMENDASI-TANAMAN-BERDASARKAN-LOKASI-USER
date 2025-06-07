from inference import predict_crop

# Contoh input (bisa dari API atau data hasil scraping)
N = 50.0
temperature = 26.0
humidity = 75.0
ph = 6.5
rainfall = 120.0

crop, confidence = predict_crop(N, temperature, humidity, ph, rainfall)
print(f"Tanaman yang direkomendasikan: {crop} ({confidence:.2f}%)")