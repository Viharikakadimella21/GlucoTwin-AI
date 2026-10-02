import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load data
df = pd.read_csv("data/HUPA0001P.csv", sep=";")
df["time"] = pd.to_datetime(df["time"])

df = df.sort_values("time").reset_index(drop=True)

# ==============================
# FEATURE ENGINEERING
# ==============================

# Previous glucose values
df["glucose_5min"] = df["glucose"].shift(1)
df["glucose_10min"] = df["glucose"].shift(2)
df["glucose_15min"] = df["glucose"].shift(3)
df["glucose_30min"] = df["glucose"].shift(6)
df["glucose_60min"] = df["glucose"].shift(12)

# Glucose trend
df["glucose_trend_30min"] = (
    df["glucose"] - df["glucose_30min"]
)

# Rolling averages
df["glucose_avg_30min"] = (
    df["glucose"].rolling(6).mean()
)

df["glucose_avg_60min"] = (
    df["glucose"].rolling(12).mean()
)

# 2-hour future glucose
future_steps = 24

df["glucose_2hr_ahead"] = (
    df["glucose"].shift(-future_steps)
)

# Remove missing values
df = df.dropna().reset_index(drop=True)

# ==============================
# INPUT FEATURES
# ==============================

features = [
    "glucose",
    "calories",
    "heart_rate",
    "steps",
    "basal_rate",
    "bolus_volume_delivered",
    "carb_input",
    "glucose_5min",
    "glucose_10min",
    "glucose_15min",
    "glucose_30min",
    "glucose_60min",
    "glucose_trend_30min",
    "glucose_avg_30min",
    "glucose_avg_60min"
]

X = df[features]
y = df["glucose_2hr_ahead"]

# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==============================
# RANDOM FOREST
# ==============================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

# ==============================
# PREDICTION
# ==============================

y_pred = model.predict(X_test)

# ==============================
# EVALUATION
# ==============================

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print()
print("==========================================")
print("   GLUCOTWIN AI - IMPROVED MODEL")
print("==========================================")

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))

print()
print("Model: Random Forest Regressor")

print()
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 4))

print("==========================================")

# ==============================
# SAVE MODEL
# ==============================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/glucose_model.pkl")

print()
print("Improved model saved successfully!")
print("Location: models/glucose_model.pkl")