import pandas as pd
import glob
import os

# Original HUPA Preprocessed folder
dataset_path = r"C:\Users\HP\Downloads\HUPA-UCM Diabetes Dataset\HUPA-UCM Diabetes Dataset\Preprocessed"

# Find all patient CSV files
files = glob.glob(os.path.join(dataset_path, "HUPA*.csv"))

print("====================================")
print("   GLUCOTWIN AI - MULTI PATIENT DATA")
print("====================================")
print("Patient files found:", len(files))

all_data = []

for file in files:
    patient_id = os.path.basename(file).replace(".csv", "")

    df = pd.read_csv(file, sep=";")
    df["patient_id"] = patient_id

    all_data.append(df)

# Combine all patients
combined_df = pd.concat(all_data, ignore_index=True)

print("Total records:", len(combined_df))
print("Total patients:", combined_df["patient_id"].nunique())

print("\nPatients:")
print(combined_df["patient_id"].unique())

print("\nColumns:")
print(combined_df.columns.tolist())

print("\nFirst 5 records:")
print(combined_df.head().to_string())

# Save combined dataset inside project
combined_df.to_csv("data/all_patients.csv", index=False)

print("\nSaved successfully:")
print("data/all_patients.csv")
print("====================================")