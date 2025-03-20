import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample dataset (Marks vs Pass/Fail)
data = {
    'marks': [35, 40, 50, 60, 70, 80, 90, 20, 30, 45, 55, 65, 75],
    'pass_fail': [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Split data
X = df[['marks']]
y = df['pass_fail']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
