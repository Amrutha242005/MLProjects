from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler safely
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"Error loading files: {e}")
    model, scaler = None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return "Model or scaler not loaded properly.", 500
    
    # Get form values
    try:
        features = [
            float(request.form['SquareFootage']),
            float(request.form['NumberOfRooms']),
            float(request.form['LotSize']),
            float(request.form['AverageDistanceToLocalAmenities']),
            float(request.form['RecentSalesPrices']),
            float(request.form['AgeOfProperty']),
            float(request.form['NumberOfFloors']),
            int(request.form['WaterSources']),
            int(request.form['Parking'])
        ]
        
        # Transform inputs using scaler
        scaled_features = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        prediction = round(prediction, 2)
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return f"Error making prediction: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)