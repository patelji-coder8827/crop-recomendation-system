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

# Navigation Menu
menu = ["Home", "Login", "Upload", "Preview"]
choice = st.sidebar.selectbox("Select Option", menu)

if choice == "Home":
    st.header("Welcome to Crop Recommendation System")

elif choice == "Upload":
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

        # Display crop images
        crop_images = {
            "rice": "images (1).jpeg",
            "apple": "apple/images (2).jpeg",
            "papaya": "apple/papaya/download (1).jpeg",
            "banana": "banana/hq720.jpg",
            "orange": "orange/desktop-wallpaper-orange-tree.jpg",
            "coconut": "coconut/coconut-farm-plantation-tree-112557128.webp",
            "cotton": "cotton/cotton-field-pictures-l79ewz6ozfd4t2l9.jpg",
            "jute": "jute/pngtree-green-jute-plantation-field-golden-fiber-green-fiber-photo-image_49156966.jpg",
            "coffee": "coffee/download.jpeg",
            "maize": "maize/different-corn-plantation-524311 (1).webp",
            "chickpea": "chickpea/istockphoto-1161019365-612x612.jpg",
            "kidneybeans": "kidneaybeans/pngtree-kidney-beans-trying-to-grow-picture-image_1729585.jpg",
            "pigeonpeas": "pigeonpeas/Fertilization-requirement-for-Pigeon-Pea-.jpg",
            "mothbeans": "mothbeans/1c7ab-photogrid_1618421436017255b1255d.jpg",
            "mungbean": "mungbeans/cover-1585090879.jpg",
            "blackgram": "blackgram/Comp1-25.jpg",
            "lentil": "lentil/5699e3e16fde8331198394.jpg",
            "pomegranate": "pomegranate/A-view-of-a-pomegranate-tree.png",
            "mango": "mango/360_F_561960690_uCMNRrqahIsdrOeEG7Lx5DzLPCof6GNe.jpg",
            "grapes": "grapesh/green-sultana-seedless-grapes-plant-1000x1000.webp",
            "watermelon": "watermalon/BONNIE_watermelon_iStock-181067852-1800px_28032150-26a6-4cda-be5b-c4408112e3a6.jpg",
            "muskmelon": "muskmelon/61b4R3CXKOL._AC_UF1000,1000_QL80_.jpg"
        }
        if recommended_crop.lower() in crop_images:
            st.image(crop_images[recommended_crop.lower()], caption=f"Recommended Crop: {recommended_crop}", use_container_width=True)

elif choice == "Login":
    st.subheader("Login Here")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        st.success("Logged in successfully")

elif choice == "Preview":
    st.subheader("Preview Data")
    st.write("Preview feature will be added soon.")

