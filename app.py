import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
import os
from PIL import Image

# Load model CNN
model = load_model('models/model_autism.h5') # Ganti dengan nama file modelmu

st.title("Deteksi Autisme pada Anak Melalui Gambar Wajah")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar wajah anak (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar yang diupload', use_column_width=True)

    # Preprocessing gambar sesuai kebutuhan model
    img = img.resize((224, 224))  # Sesuaikan ukuran input modelmu
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi jika model perlu

    # Prediksi
    pred = model.predict(img_array)
    # Asumsikan output model 0 atau 1
    if pred[0][0] > 0.5:
        st.success("Hasil Prediksi: Anak TERDETeksi Autisme")
    else:
        st.info("Hasil Prediksi: Anak TIDAK terdeteksi Autisme")
