from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['Age'])
        income = int(request.form['Income'])
        credit_score = int(request.form['Credit_Score'])
        loan_amount = int(request.form['Loan_Amount'])
        employment_status = request.form['Employment_Status']
        loan_term = int(request.form['Loan_Term'])

        # Encode Employment Status
        employment_encoded = [0, 0]  # Default: [Unemployed, Self-Employed]
        if employment_status == "Employed":
            employment_encoded = [1, 0]
        elif employment_status == "Self-Employed":
            employment_encoded = [0, 1]

        # Prepare input data for model
        input_data = np.array([[age, income, credit_score, loan_amount, loan_term] + employment_encoded])

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Approved" if prediction == 1 else "Rejected"

        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return f"Error: {str(e)}", 400  # Return error message with 400 status

if __name__ == '__main__':
    app.run(debug=True)
