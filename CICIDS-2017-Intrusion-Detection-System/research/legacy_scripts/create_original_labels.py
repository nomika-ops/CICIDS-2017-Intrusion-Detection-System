import pandas as pd
import os

base_dir = r"D:\IDS research\MachineLearningCVE"

all_files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-WorkingHours.pcap_ISCX.csv"
]

data_list = []

for file in all_files:
    filepath = os.path.join(base_dir, file)
    if os.path.exists(filepath):
        print(f"Loading: {filepath}")
        df = pd.read_csv(filepath)
        data_list.append(df)
    else:
        print(f"File not found: {filepath}")

data = pd.concat(data_list, ignore_index=True)
print(f"\nRaw shape: {data.shape}")
print(f"Total columns: {len(data.columns)}")

missing = data.isnull().sum()
print(f"\nMissing values before drop: {missing.sum()}")

data = data.dropna()
print(f"Cleaned shape: {data.shape}")

# The column name is " Label" (with a space)
label_col = " Label"

print(f"\nUsing label column: {label_col}")
print(f"Original Label distribution:")
print(data[label_col].value_counts())

output_path = r"D:\IDS research\cleaned_cicids2017_original_labels.csv"
data.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")