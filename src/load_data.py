import pandas as pd

# Load HUPA-UCM patient data
file_path = "data/HUPA0001P.csv"

df = pd.read_csv(file_path, sep=";")

# Convert time column to datetime
df["time"] = pd.to_datetime(df["time"])

print("====================================")
print("       GLUCOTWIN AI - DATA")
print("====================================")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Records:")
print(df.head().to_string())

print("\nData Types:")
print(df.dtypes)

print("====================================")