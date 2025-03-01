
# Electric Motor Temperature Prediction

## Overview
This project predicts the **Permanent Magnet Tooth Temperature (pm)** of a **Permanent Magnet Synchronous Motor (PMSM)** based on various motor parameters such as voltage, current, speed, and ambient temperature. 

## Project Structure
```
├── sample_data.csv          # Sample dataset
├── train_model.py           # Trains the ML model and saves it as pm_temperature_model.pkl
├── predict_temperature.py   # Loads model and predicts PM temperature
├── app.py                   # Flask API for real-time predictions
├── pm_temperature_model.pkl # Trained model file
├── README.md                # Project documentation
```

## Installation
Make sure you have **Python 3.x** installed. Then, install the required libraries:
```sh
pip install pandas numpy scikit-learn joblib flask
```

## Usage
### 1️⃣ Train the Model
Run the following command to train the model:
```sh
python train_model.py
```
This will generate `pm_temperature_model.pkl`.

### 2️⃣ Predict Using Script
Run:
```sh
python predict_temperature.py
```
This will output a predicted PM temperature for sample input values.

### 3️⃣ Run Flask API for Real-Time Predictions
To start the API:
```sh
python app.py
```
It will run on `http://127.0.0.1:5000/`.

To make a POST request:
```sh
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
-d '{"u_q": 300.5, "coolant": 60.2, "u_d": 220.1, "motor_speed": 3500, "i_d": 40.0, "i_q": 50.0, "ambient": 25.0}'
```

## License
This project is for educational purposes only.

---
**Author:** Gagandeepkaur Saluja 
**GitHub Repository:**[(https://github.com/Gagandeepkaur02/Electric-motor-temperature-prediction/)]
