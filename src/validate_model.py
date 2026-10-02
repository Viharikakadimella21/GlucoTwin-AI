import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/HUPA0001P.csv", sep=";")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# Create features
df["glucose_5min"] = df["glucose"].shift(1)
df["glucose_10min"] = df["glucose"].shift(2)
df["glucose_15min"] = df["glucose"].shift(3)
df["glucose_30min"] = df["glucose"].shift(6)
df["glucose_60min"] = df["glucose"].shift(12)

df["glucose_trend_30min"] = df["glucose"] - df["glucose_30min"]
df["glucose_avg_30min"] = df["glucose"].rolling(6).mean()
df["glucose_avg_60min"] = df["glucose"].rolling(12).mean()

# 2-hour future glucose
df["glucose_2hr_ahead"] = df["glucose"].shift(-24)

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

# Chronological 80/20 split
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

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("====================================")
print("   GLUCOTWIN AI - TIME VALIDATION")
print("====================================")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print()
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))
print("====================================")Get-ChildItem data\
c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0001P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0002P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0003P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0004P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0005P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0006P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0007P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0009P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0010P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0011P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0014P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0015P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0016P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0017P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0018P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0019P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0020P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0021P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0022P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0023P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0024P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0025P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0026P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0027P.csv c:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0028P.csv