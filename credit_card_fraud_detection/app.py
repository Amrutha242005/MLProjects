from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from form
        features = [
            float(request.form['transaction_amount']),
            float(request.form['time_since_last_txn']),
            float(request.form['txn_distance']),
            float(request.form['txn_count_last_hour']),
            float(request.form['avg_txn_amount_10']),
            int(request.form['fraud_history']),
            float(request.form['latitude']),
            float(request.form['longitude'])
        ]

        # Scale the input data
        features_scaled = scaler.transform([features])

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        result = "Fraudulent Transaction Detected" if prediction == 1 else "Legitimate Transaction"

        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)