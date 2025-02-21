import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("pm_temperature_model.pkl")

# Function to make a prediction
def predict_pm_temperature(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return prediction[0]

# Example input data (replace with actual values)
new_data = {
    "u_q": 300.5,
    "coolant": 60.2,
    "u_d": 220.1,
    "motor_speed": 3500,
    "i_d": 40.0,
    "i_q": 50.0,
    "ambient": 25.0
}

# Predict
predicted_temp = predict_pm_temperature(new_data)
print(f"Predicted Permanent Magnet Temperature: {predicted_temp:.2f}°C")
