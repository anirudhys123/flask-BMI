import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Sample dataset
data = pd.DataFrame({
    'age': [22, 30, 45, 25, 35, 60, 50, 28, 40, 55],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male'],
    'bmi': [19.5, 22.3, 28.0, 25.1, 30.2, 32.5, 26.8, 21.5, 24.3, 33.5],
    'risk': ['Healthy', 'Healthy', 'Moderate Risk', 'Healthy', 'Moderate Risk', 'High Risk', 'Moderate Risk', 'Healthy', 'Healthy', 'High Risk']
})

# Encode gender and risk
le_gender = LabelEncoder()
le_risk = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['gender'])
data['risk'] = le_risk.fit_transform(data['risk'])  # 0 = Healthy, 1 = High Risk, 2 = Moderate Risk

# Train model
X = data[['age', 'gender', 'bmi']]
y = data['risk']
# Increase max_iter from default (100) to 1000
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save the model and encoders
joblib.dump(model, 'risk_model.pkl')
joblib.dump(le_gender, 'gender_encoder.pkl')
joblib.dump(le_risk, 'risk_encoder.pkl')
