from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoder
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return "✅ Mental Health API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(df)[0]
    label = le.inverse_transform([prediction])[0]

    # Suggestions
    suggestions = []

    if data.get("Daily_Sleep_Hours", 8) < 6:
        suggestions.append("Improve sleep (7-8 hours recommended)")

    if data.get("Screen_Time_Hours", 0) > 8:
        suggestions.append("Reduce screen time")

    if data.get("Exercise_Frequency_per_Week", 3) < 2:
        suggestions.append("Increase physical activity")

    if data.get("Overthinking_Score", 0) > 7:
        suggestions.append("Practice mindfulness")

    return jsonify({
        "Burnout_Risk": label,
        "Suggestions": suggestions
    })

if __name__ == '__main__':
    app.run(debug=True)