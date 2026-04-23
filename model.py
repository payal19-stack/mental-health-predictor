
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\mental_health_project\genz_mental_wellness_synthetic_dataset.csv")

# Check columns
print(df.columns)

# Select only required 4 features
X = df[
    [
        "Daily_Sleep_Hours",
        "Screen_Time_Hours",
        "Exercise_Frequency_per_Week",
        "Overthinking_Score"
    ]
]

# Target column
y = df["Burnout_Risk"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y_encoded)

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ New model trained successfully!")
print("✅ model.pkl created")
print("✅ label_encoder.pkl created")