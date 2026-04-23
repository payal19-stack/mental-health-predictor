import streamlit as st
import joblib
import pandas as pd

# Load trained model and label encoder
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# Page configuration
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 Student Mental Health Risk Predictor")
st.write("Enter your daily wellness details to predict burnout risk and get suggestions.")

# Input fields
sleep_hours = st.number_input(
    "Daily Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.5
)

screen_time = st.number_input(
    "Screen Time Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

exercise_frequency = st.number_input(
    "Exercise Frequency per Week",
    min_value=0,
    max_value=14,
    value=3,
    step=1
)

overthinking_score = st.slider(
    "Overthinking Score",
    min_value=1,
    max_value=10,
    value=5
)

# Prediction button
if st.button("Predict Burnout Risk"):

    # Create DataFrame (must match training columns exactly)
    input_data = pd.DataFrame({
        "Daily_Sleep_Hours": [sleep_hours],
        "Screen_Time_Hours": [screen_time],
        "Exercise_Frequency_per_Week": [exercise_frequency],
        "Overthinking_Score": [overthinking_score]
    })

    # Predict
    prediction = model.predict(input_data)[0]
    result = le.inverse_transform([prediction])[0]

    # Suggestions logic
    suggestions = []

    if sleep_hours < 6:
        suggestions.append("Improve sleep duration (recommended: 7–8 hours daily)")

    if screen_time > 8:
        suggestions.append("Reduce excessive screen time")

    if exercise_frequency < 2:
        suggestions.append("Increase physical activity and regular exercise")

    if overthinking_score > 7:
        suggestions.append("Practice mindfulness, meditation, and stress management")

    # Display result
    st.subheader("Prediction Result")
    st.success(f"Burnout Risk Level: {result}")

    # Display suggestions
    st.subheader("Suggestions for Better Mental Health")

    if suggestions:
        for item in suggestions:
            st.write(f"• {item}")
    else:
        st.write("No major risk indicators detected. Keep maintaining a healthy routine.")

# Footer
st.markdown("---")
st.caption("Built using Python, Pandas, Scikit-learn, and Streamlit")