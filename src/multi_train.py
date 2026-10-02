import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load all patients
df = pd.read_csv("data/all_patients.csv")

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values(["patient_id", "time"]).reset_index(drop=True)

# Create time-series features separately for each patient
group = df.groupby("patient_id")

df["glucose_5min"] = group["glucose"].shift(1)
df["glucose_10min"] = group["glucose"].shift(2)
df["glucose_15min"] = group["glucose"].shift(3)
df["glucose_30min"] = group["glucose"].shift(6)
df["glucose_60min"] = group["glucose"].shift(12)

df["glucose_trend_30min"] = (
    df["glucose"] - df["glucose_30min"]
)

df["glucose_avg_30min"] = (
    group["glucose"].transform(lambda x: x.rolling(6).mean())
)

df["glucose_avg_60min"] = (
    group["glucose"].transform(lambda x: x.rolling(12).mean())
)

# 2-hour future glucose
df["glucose_2hr_ahead"] = group["glucose"].shift(-24)

df = df.dropna().reset_index(drop=True)

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

# 80/20 chronological split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("====================================")
print(" GLUCOTWIN AI - MULTI PATIENT MODEL")
print("====================================")
print("Patients:", df["patient_id"].nunique())
print("Total records:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print()
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))
print("====================================")

# Save model
joblib.dump(model, "models/multi_patient_model.pkl")

print("Model saved:")
print("models/multi_patient_model.pkl")