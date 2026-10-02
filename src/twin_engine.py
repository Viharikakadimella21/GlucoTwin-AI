import pandas as pd
import joblib

MODEL_PATH = "models/glucose_model.pkl"
DATA_PATH = "data/HUPA0001P.csv"

model = joblib.load(MODEL_PATH)


FEATURES = [
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


def prepare_data():

    df = pd.read_csv(DATA_PATH, sep=";")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Lag features
    df["glucose_5min"] = df["glucose"].shift(1)
    df["glucose_10min"] = df["glucose"].shift(2)
    df["glucose_15min"] = df["glucose"].shift(3)
    df["glucose_30min"] = df["glucose"].shift(6)
    df["glucose_60min"] = df["glucose"].shift(12)

    # Trend
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

    return df.dropna().reset_index(drop=True)


def predict_glucose(patient_state):

    input_data = pd.DataFrame(
        [patient_state],
        columns=FEATURES
    )

    return model.predict(input_data)[0]


def risk_level(glucose):

    if glucose < 70:
        return "LOW GLUCOSE RISK"

    elif glucose >= 180:
        return "HIGH GLUCOSE RISK"

    else:
        return "NORMAL RANGE"


def what_if_simulation(patient_state, carb_change=0, step_change=0):

    scenario = patient_state.copy()

    # Change meal carbohydrate
    scenario["carb_input"] = max(
        0,
        scenario["carb_input"] + carb_change
    )

    # Change physical activity
    scenario["steps"] = max(
        0,
        scenario["steps"] + step_change
    )

    predicted = predict_glucose(scenario)

    return predicted


if __name__ == "__main__":

    df = prepare_data()

    # Select current patient state
    current_state = df.iloc[100][FEATURES].to_dict()

    # Current prediction
    current_prediction = predict_glucose(current_state)

    print("==========================================")
    print("       GLUCOTWIN AI")
    print("       WHAT-IF SIMULATION")
    print("==========================================")

    print()
    print(
        "Current predicted glucose:",
        round(current_prediction, 2)
    )

    # Scenario 1: More carbohydrates
    high_carb_prediction = what_if_simulation(
        current_state,
        carb_change=30,
        step_change=0
    )

    print()
    print(
        "After +30g carbs:",
        round(high_carb_prediction, 2)
    )

    # Scenario 2: More physical activity
    activity_prediction = what_if_simulation(
        current_state,
        carb_change=0,
        step_change=500
    )

    print()
    print(
        "After +500 steps:",
        round(activity_prediction, 2)
    )

    print()
    print("Risk - Current:",
          risk_level(current_prediction))

    print("Risk - +30g carbs:",
          risk_level(high_carb_prediction))

    print("Risk - +500 steps:",
          risk_level(activity_prediction))

    print("==========================================")