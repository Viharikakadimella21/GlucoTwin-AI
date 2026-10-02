import pandas as pd

df = pd.read_csv("data/all_patients.csv")

profiles = []

for patient_id, patient_data in df.groupby("patient_id"):

    profile = {
        "patient_id": patient_id,
        "average_glucose": round(patient_data["glucose"].mean(), 2),
        "minimum_glucose": round(patient_data["glucose"].min(), 2),
        "maximum_glucose": round(patient_data["glucose"].max(), 2),
        "average_heart_rate": round(patient_data["heart_rate"].mean(), 2),
        "average_steps": round(patient_data["steps"].mean(), 2),
        "total_carbs": round(patient_data["carb_input"].sum(), 2),
        "total_insulin": round(
            patient_data["bolus_volume_delivered"].sum(), 2
        )
    }

    profiles.append(profile)

profile_df = pd.DataFrame(profiles)

profile_df.to_csv("data/patient_profiles.csv", index=False)

print("====================================")
print(" GLUCOTWIN AI - PATIENT PROFILES")
print("====================================")
print("Patients:", len(profile_df))
print()
print(profile_df.head().to_string(index=False))
print()
print("Saved successfully:")
print("data/patient_profiles.csv")
print("====================================")