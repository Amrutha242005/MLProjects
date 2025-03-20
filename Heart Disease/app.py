from flask import Flask, render_template, request
import numpy as np
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
        user_input = [float(request.form[key]) for key in request.form.keys()]
        
        # Scale the input data
        user_input_scaled = scaler.transform([user_input])
        
        # Make prediction
        prediction = model.predict(user_input_scaled)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        
        return render_template("result.html", result=result)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
