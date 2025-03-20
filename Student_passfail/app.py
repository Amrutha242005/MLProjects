from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        marks = float(request.form['marks'])
        prediction = model.predict(np.array([[marks]]))
        result = "Pass" if prediction[0] == 1 else "Fail"
        return render_template('index.html', result=result)
    except:
        return render_template('index.html', result="Invalid Input")

if __name__ == '__main__':
    app.run(debug=True)
