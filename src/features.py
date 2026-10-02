import pandas as pd

file_path = "data/HUPA0001P.csv"

df = pd.read_csv(file_path, sep=";")
df["time"] = pd.to_datetime(df["time"])

df = df.sort_values("time").reset_index(drop=True)

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
    df["glucose"].rolling(window=6).mean()
)

df["glucose_avg_60min"] = (
    df["glucose"].rolling(window=12).mean()
)

# 2-hour future glucose
future_steps = 24

df["glucose_2hr_ahead"] = (
    df["glucose"].shift(-future_steps)
)

# Remove rows with missing values
df = df.dropna().reset_index(drop=True)

print("==========================================")
print("      GLUCOTWIN AI - FEATURE ENGINE")
print("==========================================")

print("Rows after feature engineering:", len(df))

print("\nNew Features:")
print([
    "glucose_5min",
    "glucose_10min",
    "glucose_15min",
    "glucose_30min",
    "glucose_60min",
    "glucose_trend_30min",
    "glucose_avg_30min",
    "glucose_avg_60min"
])

print("\nSample data:")
print(
    df[
        [
            "glucose",
            "glucose_30min",
            "glucose_60min",
            "glucose_trend_30min",
            "glucose_2hr_ahead"
        ]
    ].head()
)

print("==========================================")