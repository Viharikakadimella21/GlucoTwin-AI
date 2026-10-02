import pandas as pd
import joblib

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


def predict(patient_state):
    input_data = patient_state[features].to_frame().T
    return model.predict(input_data)[0]


# Select patient
patient_id = df["patient_id"].iloc[0]

patient_df = df[df["patient_id"] == patient_id]
patient_df = create_features(patient_df).dropna()

current = patient_df.iloc[-1].copy()

# Current prediction
current_prediction = predict(current)

# Scenario 1: more carbohydrates
carb_scenario = current.copy()
carb_scenario["carb_input"] += 30
carb_prediction = predict(carb_scenario)

# Scenario 2: more activity
activity_scenario = current.copy()
activity_scenario["steps"] += 500
activity_prediction = predict(activity_scenario)

print("====================================")
print(" GLUCOTWIN AI - WHAT-IF SIMULATION")
print("====================================")

print("Patient:", patient_id)

print()
print("CURRENT STATE")
print("Predicted glucose:", round(current_prediction, 2))

print()
print("WHAT IF +30g CARBOHYDRATES?")
print("Predicted glucose:", round(carb_prediction, 2))

print()
print("WHAT IF +500 STEPS?")
print("Predicted glucose:", round(activity_prediction, 2))

print("====================================")