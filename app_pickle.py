import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

# Load Model dan Scaler
@st.cache_resource
def load_model():
    with open('mlp_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_scaler():
    with open('mlp_model.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

model = load_model()
scaler = load_scaler()

# Sidebar navigation
st.sidebar.title("Breast Cancer Prediction")
navigation = st.sidebar.radio("Navigation", ["Data Understanding", "Prediction"])

# Section: Data Understanding
if navigation == "Data Understanding":
    st.title("Data Understanding")
    st.write("Di bagian ini, Anda dapat memahami fitur-fitur yang digunakan dalam model prediksi kanker payudara.")
    
    st.write("""
    **Penjelasan fitur:**
    - **Mean Radius**: Rata-rata jari-jari sel tumor.
    - **Mean Texture**: Tekstur rata-rata dari sel tumor.
    - **Mean Perimeter**: Keliling rata-rata dari sel tumor.
    - **Mean Area**: Luas rata-rata dari sel tumor.
    """)

    st.write("Semua fitur ini telah di-scaling menggunakan StandardScaler untuk memastikan skala data konsisten sebelum digunakan dalam model.")

# Section: Prediction
elif navigation == "Prediction":
    st.title("Prediction")
    st.write("Masukkan nilai fitur untuk memprediksi apakah tumor adalah kanker jinak atau ganas.")

    # Input features for prediction
    radius_mean = st.number_input("Mean Radius", min_value=0.0, max_value=30.0, step=0.1)
    texture_mean = st.number_input("Mean Texture", min_value=0.0, max_value=40.0, step=0.1)
    perimeter_mean = st.number_input("Mean Perimeter", min_value=0.0, max_value=200.0, step=0.1)
    area_mean = st.number_input("Mean Area", min_value=0.0, max_value=2500.0, step=1.0)

    # Prepare data for prediction
    input_data = pd.DataFrame([[radius_mean, texture_mean, perimeter_mean, area_mean]],
                              columns=["Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area"])
    scaled_input = scaler.transform(input_data)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(scaled_input)
        if prediction[0] == 0:
            st.success("Prediksi: Tumor Jinak")
        else:
            st.error("Prediksi: Tumor Ganas")
