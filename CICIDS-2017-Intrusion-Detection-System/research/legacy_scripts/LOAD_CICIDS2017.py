import pandas as pd
import os
import numpy as np

folder = r"D:\IDS research\MachineLearningCVE"

files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

dfs = []
for f in files:
    path = os.path.join(folder, f)
    if os.path.exists(path):
        print("Loading:", path)
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
    else:
        print("Missing:", path)

data = pd.concat(dfs, ignore_index=True)
data.columns = data.columns.str.strip()

print("Raw shape:", data.shape)
print("Columns:", len(data.columns))
print("Label column present:", "Label" in data.columns)

data = data.replace([np.inf, -np.inf], np.nan)
print("Missing values before drop:", data.isna().sum().sum())

data = data.drop_duplicates()
data = data.dropna()

print("Cleaned shape:", data.shape)
print("Remaining missing values:", data.isna().sum().sum())
print("\nOriginal Label Distribution:\n")
print("\nNumber of Classes:", data["Label"].nunique())

# Save original multiclass dataset
data.to_csv(
    r"D:\IDS research\cleaned_cicids2017_multiclass.csv",
    index=False
)

print("\nSaved:")
print(r"D:\IDS research\cleaned_cicids2017_multiclass.csv")