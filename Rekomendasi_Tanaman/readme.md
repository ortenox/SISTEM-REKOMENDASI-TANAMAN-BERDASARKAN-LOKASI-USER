jalankan di terminal

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

streamlit run app.py

Browser akan otomatis terbuka ke alamat localhost (contohnya http://localhost:8501)
