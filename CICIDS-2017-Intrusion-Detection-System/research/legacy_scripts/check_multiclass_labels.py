import pandas as pd

df = pd.read_csv(r"D:\IDS research\cleaned_cicids2017_original_labels.csv")

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

print("Number of Classes:", df["Label"].nunique())
print("\nClass Distribution:\n")
print(df["Label"].value_counts())