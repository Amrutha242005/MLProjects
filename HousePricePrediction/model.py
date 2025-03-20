import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load data
data = pd.read_csv('house_data.csv')

# Features and target
X = data[['Square Footage', 'Number of Rooms', 'Lot Size', 'Average Distance to Local Amenities', 
          'Recent Sales Prices of Similar Homes (Lakhs)', 'Age of the Property', 'Number of Floors', 
          'Water_Sources', 'Parking']]
y = data['Price (Lakhs)']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler have been saved.")
