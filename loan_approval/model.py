import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('loan_data.csv')

# Print column names to verify
print("Columns in dataset:", df.columns)

# Remove spaces or unexpected characters in column names
df.columns = df.columns.str.strip()

# Verify the actual target column name
target_column = 'Loan_Status'  # Update this if needed

# Check if target column exists
if target_column not in df.columns:
    raise KeyError(f"Column '{target_column}' not found in dataset. Available columns: {df.columns}")

# Convert categorical target variable to numerical (if applicable)
df[target_column] = df[target_column].map({'Approved': 1, 'Rejected': 0})

# Define features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert categorical features to numerical (if any exist)
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

# Evaluate model performance
y_pred = rf_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

