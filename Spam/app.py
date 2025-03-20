from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pre-trained models and vectorizer
log_reg_model = pickle.load(open('log_reg_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email']
        email_vec = vectorizer.transform([email_text])

        # Predict with Logistic Regression model
        log_reg_pred = log_reg_model.predict(email_vec)[0]
        
        # Predict with k-NN model
        knn_pred = knn_model.predict(email_vec)[0]

        return render_template('result.html', log_reg_pred=log_reg_pred, knn_pred=knn_pred)

if __name__ == "__main__":
    app.run(debug=True)
