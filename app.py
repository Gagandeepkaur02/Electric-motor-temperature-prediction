from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("pm_temperature_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(df)[0]
        return jsonify({"predicted_pm_temperature": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
