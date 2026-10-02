import pandas as pd
import joblib

# Load data and model
df = pd.read_csv("data/all_patients.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values(["patient_id", "time"]).reset_index(drop=True)

model = joblib.load("models/multi_patient_model.pkl")


def create_features(patient_df):
    patient_df = patient_df.copy()
    patient_df = patient_df.sort_values("time").reset_index(drop=True)

    patient_df["glucose_5min"] = patient_df["glucose"].shift(1)
    patient_df["glucose_10min"] = patient_df["glucose"].shift(2)
    patient_df["glucose_15min"] = patient_df["glucose"].shift(3)
    patient_df["glucose_30min"] = patient_df["glucose"].shift(6)
    patient_df["glucose_60min"] = patient_df["glucose"].shift(12)

    patient_df["glucose_trend_30min"] = (
        patient_df["glucose"] - patient_df["glucose_30min"]
    )

    patient_df["glucose_avg_30min"] = (
        patient_df["glucose"].rolling(6).mean()
    )

    patient_df["glucose_avg_60min"] = (
        patient_df["glucose"].rolling(12).mean()
    )

    return patient_df


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


def risk_level(glucose):

    if glucose < 70:
        return "LOW GLUCOSE RISK"

    elif glucose >= 180:
        return "HIGH GLUCOSE RISK"

    else:
        return "NORMAL RANGE"


def predict_patient(patient_id):

    patient_df = df[df["patient_id"] == patient_id]

    patient_df = create_features(patient_df)
    patient_df = patient_df.dropna()

    latest = patient_df.iloc[-1]

    input_data = latest[features].to_frame().T

    prediction = model.predict(input_data)[0]

    return latest, prediction


# Test patient
patient_id = df["patient_id"].iloc[0]

latest, prediction = predict_patient(patient_id)

print("====================================")
print("     GLUCOTWIN AI - DIGITAL TWIN")
print("====================================")

print("Patient:", patient_id)
print("Current glucose:", latest["glucose"])
print("Heart rate:", latest["heart_rate"])
print("Steps:", latest["steps"])
print("Carbohydrates:", latest["carb_input"])

print()
print("Predicted glucose after 2 hours:",
      round(prediction, 2))

print("Risk:", risk_level(prediction))

print("====================================")