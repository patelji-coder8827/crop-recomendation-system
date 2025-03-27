import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Load pre-trained models and encoders
scaler = joblib.load("scaler.pkl")
best_model = joblib.load("best_model.pkl")
# Define the prediction function
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    input_data = scaler.transform(input_data)  # Standardize input
    prediction = best_model.predict(input_data)[0]
    return prediction
# Streamlit UI
st.title("🌱 Crop Recommendation System")

st.sidebar.header("Enter Soil and Weather Parameters")

# User input fields
N = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

# Prediction button
if st.sidebar.button("Predict Crop"):
    recommended_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    st.success(f"🌾 Recommended Crop: **{recommended_crop}**")

 # Show rice image if recommended crop is rices
    if recommended_crop.lower() == "rice":
        st.image("rice.jpg", caption="Recommended Crop: Rice",  use_container_width=True)
    elif recommended_crop.lower() == "apple":
        st.image("apple/apple.jpg", caption="Recommended Crop: Apple",  use_container_width=True)
    elif recommended_crop.lower() == "papaya":
        st.image("apple/papaya/download (1).jpeg", caption="Recommended Crop:papaya",  use_container_width=True)
    elif recommended_crop.lower() == "banana":
        st.image("banana/hq720.jpg", caption="Recommended Crop:banana",  use_container_width=True)
    elif recommended_crop.lower() == "orange":
        st.image("orange/desktop-wallpaper-orange-tree.jpg", caption="Recommended Crop:orange",  use_container_width=True)
    elif recommended_crop.lower() == "coconut":
        st.image("coconut/coconut-farm-plantation-tree-112557128.webp", caption="Recommended Crop:orangecoconut",  use_container_width=True)
    elif recommended_crop.lower() == "cotton":
        st.image("cotton/cotton-field-pictures-l79ewz6ozfd4t2l9.jpg", caption="Recommended Crop:cotton",  use_container_width=True)
    elif recommended_crop.lower() == "jute":
        st.image("jute/pngtree-green-jute-plantation-field-golden-fiber-green-fiber-photo-image_49156966.jpg", caption="Recommended Crop:cotton", use_container_width=True)
    elif recommended_crop.lower() == "coffee":
        st.image("coffee/download.jpeg", caption="Recommended Crop:coffee",  use_container_width=True)
    elif recommended_crop.lower() == "maize":
        st.image("maize/different-corn-plantation-524311 (1).webp", caption="Recommended Crop:maize",  use_container_width=True)
    elif recommended_crop.lower() == "chickpea":
        st.image("chickpea/istockphoto-1161019365-612x612.jpg", caption="Recommended Crop:chickpea",  use_container_width=True)
    elif recommended_crop.lower() == "kidneybeans":
        st.image("kidneaybeans/pngtree-kidney-beans-trying-to-grow-picture-image_1729585.jpg", caption="Recommended Crop:kidneybeans", use_column_width=True)
    elif recommended_crop.lower() == "pigeonpeas":
        st.image("pigeonpeas/Fertilization-requirement-for-Pigeon-Pea-.jpg", caption="Recommended Crop:pigeonpeas",  use_container_width=True)
    elif recommended_crop.lower() == "mothbeans":
        st.image("mothbeans/1c7ab-photogrid_1618421436017255b1255d.jpg", caption="Recommended Crop:mothbeans",use_container_width=True)
    elif recommended_crop.lower() == "mungbean":
        st.image("mungbeans/cover-1585090879.jpg", caption="Recommended Crop:mungbean",use_container_width=True)
    elif recommended_crop.lower() == "blackgram":
        st.image("blackgram/Comp1-25.jpg", caption="Recommended Crop:blackgram",use_container_width=True)
    elif recommended_crop.lower() == "lentil":
        st.image("lentil/5699e3e16fde8331198394.jpg", caption="Recommended Crop:lentil",use_container_width=True)
    elif recommended_crop.lower() == "pomegranate":
        st.image("pomegranate/A-view-of-a-pomegranate-tree.png", caption="Recommended Crop:pomegranate",use_container_width=True)
    elif recommended_crop.lower() == "mango":
        st.image("mango/360_F_561960690_uCMNRrqahIsdrOeEG7Lx5DzLPCof6GNe.jpg", caption="Recommended Crop:mango",use_container_width=True)
    elif recommended_crop.lower() == "grapes":
        st.image("grapesh/green-sultana-seedless-grapes-plant-1000x1000.webp", caption="Recommended Crop:grapes",use_container_width=True)
    elif recommended_crop.lower() == "watermelon":
        st.image("watermalon/BONNIE_watermelon_iStock-181067852-1800px_28032150-26a6-4cda-be5b-c4408112e3a6.jpg", caption="Recommended Crop:watermelon",use_container_width=True)
    elif recommended_crop.lower() == "muskmelon":
        st.image("muskmelon/61b4R3CXKOL._AC_UF1000,1000_QL80_.jpg", caption="Recommended Crop:melon",use_container_width=True)
        