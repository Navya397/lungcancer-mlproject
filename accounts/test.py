# test.py

import pandas as pd
import pickle

# Load model, scaler, and training columns
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scalar.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))  # Training column names

# Create new input data (use same column names as in training)
new_data = pd.DataFrame([{
    'AGE': 55,
    'SMOKING': 1,
    'YELLOW_FINGERS': 1,
    'ANXIETY': 0,
    'PEER_PRESSURE': 1,
    'CHRONIC_DISEASE': 1,
    'FATIGUE': 1,
    'ALLERGY': 0,
    'WHEEZING': 1,
    'ALCOHOL_CONSUMING': 0,
    'COUGHING': 1,
    'SHORTNESS_OF_BREATH': 1,
    'SWALLOWING_DIFFICULTY': 0,
    'CHEST_PAIN': 1
}])

# Ensure all expected columns are present
for col in columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Drop unexpected columns and reorder
new_data = new_data[columns]

# Convert to NumPy array before passing to scaler (avoid feature name checks)
new_data_scaled = scaler.transform(new_data.to_numpy())

# Predict
prediction = model.predict(new_data_scaled)
print("Prediction (0 = No Cancer, 1 = Cancer):", prediction[0])