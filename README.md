# 🌾 FarmAI - Crop Recommendation System

**FarmAI** is an intelligent machine learning-based crop recommendation system designed to assist farmers in selecting the most suitable crops based on environmental, economic, and soil-related factors.

---

## 🚀 Problem Statement

Farmers face difficulties in choosing the right crop due to fluctuating market conditions, climate change, and soil diversity. FarmAI aims to simplify this process by analyzing:
- Soil condition
- Market trends
- Climate impact

And recommending the **top 3 most profitable and suitable crops**.

---

## 🧠 Proposed Solution

We trained a supervised machine learning model using real-world datasets covering 18 features, such as:
- Soil pH, Moisture, Temperature, Rainfall
- Market Price, Supply & Demand Index
- Fertilizer & Pesticide Usage
- Government subsidies, water depth, and more

The model provides accurate crop recommendations using a trained neural network (`.h5`) and preprocessed with StandardScaler and Encoders (`.pkl` files).

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas, NumPy**
- **Jupyter Notebook / Python scripts**
- **Joblib** (for saving models)

---
## Project Structure
FarmAI: An epert everyone need │ 
├── crop_predictor.py # Main prediction script 
├── crop_prediction_model.h5 # Trained Keras model 
├── scaler.pkl # StandardScaler for feature scaling 
├── label_encoder.pkl # LabelEncoder for crop labels 
├── ordinal_encoder.pkl # OrdinalEncoder for categorical features 
├── farmer_advisor_dataset.csv # Dataset used for training  
├── Market_researcher_dataset.csv # Dataset used for training
