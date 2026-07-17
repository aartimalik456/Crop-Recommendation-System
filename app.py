import streamlit as st
import joblib
import pandas as pd

# Sidebar
st.sidebar.title("🌾 Crop Recommendation System")
st.sidebar.markdown("---")
st.sidebar.subheader("👩‍💻 Developed By")
st.sidebar.write("Aarti")
st.sidebar.write("B.Tech CSE")
st.sidebar.write("CGC Landran")
st.sidebar.write("### Project Information")
st.sidebar.write("""
This application recommends the best crop based on:
- 🌱 Nitrogen (N)
- 🌱 Phosphorus (P)
- 🌱 Potassium (K)
- 🌡 Temperature
- 💧 Humidity
- ⚗ pH
- 🌧 Rainfall
""")
st.sidebar.info("Developed using AI & Machine Learning")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.title("🌱 AI-Based Crop Recommendation System")
st.markdown("""
### Welcome!
Enter the soil nutrients and weather conditions below.
Our Machine Learning model will recommend the most suitable crop for cultivation.
""")
st.divider()
# User Input

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("🌱 Nitrogen (N)", min_value=0.0)
    P = st.number_input("🌱 Phosphorus (P)", min_value=0.0)
    K = st.number_input("🌱 Potassium (K)", min_value=0.0)
    ph = st.number_input("⚗️ pH", min_value=0.0)

with col2:
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0)
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0)
    
# Predict Button
st.button("🌾 Predict Crop")

    # Create DataFrame
input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

    # Scale the input
input_scaled = scaler.transform(input_data)
st.subheader("📋 Input Values")

st.dataframe(input_data)
    # Predict crop
with st.spinner("🔍 Analyzing soil and weather conditions..."):
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        confidence = probability.max() * 100
        
    # Display result
st.success(f"✅ Recommended Crop: **{prediction[0].capitalize()}**")
st.info(f"🎯 Model Confidence: {confidence:.2f}%")
   