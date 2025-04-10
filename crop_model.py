import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

print("Loading CSV files...")
farm_df = pd.read_csv("farmer_advisor_dataset.csv")
market_df = pd.read_csv("market_researcher_dataset.csv")

print("Merging datasets...")
df = pd.merge(farm_df, market_df, left_on="Crop_Type", right_on="Product", how="inner")
df.drop(columns=["Farm_ID", "Market_ID", "Product", "Crop_Yield_ton"], inplace=True)

df["Live_Temperature"] = 30 + np.random.normal(0, 2, size=len(df))
df["Live_Humidity"] = 60 + np.random.normal(0, 5, size=len(df))
df["Weather_Score"] = 70 + np.random.normal(0, 3, size=len(df))

print("Encoding crop labels...")
label_encoder = LabelEncoder()
df["Crop_Label"] = label_encoder.fit_transform(df["Crop_Type"])
y = to_categorical(df["Crop_Label"])

joblib.dump(label_encoder, "label_encoder.pkl")

print("Encoding other categorical columns...")
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols.remove("Crop_Type") 
ordinal_encoder = OrdinalEncoder()
df[categorical_cols] = ordinal_encoder.fit_transform(df[categorical_cols])

joblib.dump(ordinal_encoder, "ordinal_encoder.pkl")

print("Defining features...")
features = [
    "Soil_pH", "Soil_Moisture", "Temperature_C", "Rainfall_mm",
    "Fertilizer_Usage_kg", "Pesticide_Usage_kg", "Sustainability_Score",
    "Market_Price_per_ton", "Demand_Index", "Supply_Index",
    "Competitor_Price_per_ton", "Economic_Indicator", "Weather_Impact_Score",
    "Seasonal_Factor", "Consumer_Trend_Index",
    "Live_Temperature", "Live_Humidity", "Weather_Score"
]

X = df[features].values

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

print("📈 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Calculating class weights to handle imbalance...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df["Crop_Label"]),
    y=df["Crop_Label"]
)
class_weights_dict = dict(enumerate(class_weights))

print("Building deep learning model...")
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training model")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=1024,
    validation_split=0.1,
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=1
)

print("Saving model...")
model.save("crop_prediction_model.h5")

print("Training complete. Model and encoders saved!")
