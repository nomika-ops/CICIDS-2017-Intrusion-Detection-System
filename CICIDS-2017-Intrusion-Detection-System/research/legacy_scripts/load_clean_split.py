import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("="*70)
print("STEP 2 : TRAIN TEST SPLIT (BINARYCLASS)")
print("="*70)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

df = pd.read_csv(r"D:\IDS research\cleaned_cicids2017_binary.csv")
df.columns = df.columns.str.strip()

print("\nDataset Shape :", df.shape)

# ------------------------------------------------------------
# Features and Target
# ------------------------------------------------------------

X = df.drop("Label", axis=1)
y = df["Label"]

# ------------------------------------------------------------
# Remove Duplicate Feature Rows
# ------------------------------------------------------------

duplicate_mask = X.duplicated()

print(f"\nDuplicate feature rows found: {duplicate_mask.sum()}")

# Remove duplicate rows while keeping the first occurrence
df = df.loc[~duplicate_mask].reset_index(drop=True)

# Recreate X and y after removing duplicates
X = df.drop("Label", axis=1)
y = df["Label"]

print("Dataset shape after removing duplicates:", df.shape)

# ------------------------------------------------------------
# Label Encoding
# ------------------------------------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# Save Label Encoder
os.makedirs(r"D:\IDS research\outputs", exist_ok=True)

joblib.dump(
    encoder,
    r"D:\IDS research\outputs\label_encoder.pkl"
)

print("\nNumber of Classes:", len(encoder.classes_))

# ------------------------------------------------------------
# Train Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)
train_rows = set(map(tuple, X_train.values))
test_rows = set(map(tuple, X_test.values))

overlap = train_rows.intersection(test_rows)

print(f"\nDuplicate feature rows between train and test: {len(overlap)}")

assert len(overlap) == 0, "ERROR: Duplicate feature rows detected between train and test!"

# ------------------------------------------------------------
# Save Files
# ------------------------------------------------------------

X_train.to_csv(
    r"D:\IDS research\outputs\X_train.csv",
    index=False
)

X_test.to_csv(
    r"D:\IDS research\outputs\X_test.csv",
    index=False
)

pd.DataFrame(y_train, columns=["Label"]).to_csv(
    r"D:\IDS research\outputs\y_train.csv",
    index=False
)

pd.DataFrame(y_test, columns=["Label"]).to_csv(
    r"D:\IDS research\outputs\y_test.csv",
    index=False
)

print("\nTraining Label Distribution")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTesting Label Distribution")
print(pd.Series(y_test).value_counts().sort_index())

print("\nSaved Files")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")
print("label_encoder.pkl")

print("\nSTEP 2 COMPLETED SUCCESSFULLY")