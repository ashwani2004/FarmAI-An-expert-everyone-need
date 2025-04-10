import numpy as np
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model

model = load_model('crop_prediction_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

input_data = {
    'Soil_pH': 7.8,
    'Soil_Moisture': 50,
    'Temperature_C': 36,
    'Rainfall_mm': 20,
    'Market_Price_per_ton': 3000,
    'Demand_Index': 0.9,
    'Supply_Index': 0.7,
    'Competitor_Price_per_ton': 1800,
    'Economic_Indicator': 1.05,
    'Weather_Impact_Score': 0.8,
    'Seasonal_Factor': 0.9,
    'Consumer_Trend_Index': 0.95,
    'Fertilizer_Usage_Index': 0.75,
    'Irrigation_Availability': 1,
    'Pesticide_Usage_Index': 0.6,
    'Soil_Type_Score': 0.82,
    'Water_Table_Depth': 10,
    'Government_Subsidy_Index': 0.88
}

df_input = pd.DataFrame([input_data])
scaled_input = scaler.transform(df_input)
prediction = model.predict(scaled_input)
top_indices = np.argsort(prediction[0])[::-1][:3]
top_crops = label_encoder.inverse_transform(top_indices)
top_probs = prediction[0][top_indices] * 100

print("\nTop 3 Recommended Crops:")
for crop, prob in zip(top_crops, top_probs):
    print(f"{crop}: {prob:.2f}%")

conn = sqlite3.connect('crop_predictions.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS crop_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    input_data TEXT,
    top_crop_1 TEXT,
    confidence_1 REAL,
    top_crop_2 TEXT,
    confidence_2 REAL,
    top_crop_3 TEXT,
    confidence_3 REAL
)
''')
cursor.execute('''
INSERT INTO crop_predictions (
    timestamp, input_data, 
    top_crop_1, confidence_1,
    top_crop_2, confidence_2,
    top_crop_3, confidence_3
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', (
    datetime.now().isoformat(),
    df_input.to_json(),
    top_crops[0], float(top_probs[0]),
    top_crops[1], float(top_probs[1]),
    top_crops[2], float(top_probs[2])
))
cursor.execute('''SELECT * FROM crop_predictions''')
rows=cursor.fetchall()
for row in rows:
    print(row)
conn.commit()
conn.close()
print("\nPrediction saved to crop_predictions.db")
