import pandas as pd
import numpy as np
import os
import joblib

# ============================================================
# STEP 6 : SELECTIVE LOG TRANSFORMATION
# ============================================================

print("=" * 70)
print("STEP 6 : SELECTIVE LOG TRANSFORMATION")
print("=" * 70)

INPUT_FOLDER = r"D:\IDS research\outputs"
OUTPUT_FOLDER = r"D:\IDS research\outputs"

# ------------------------------------------------------------
# Load Train/Test Data
# ------------------------------------------------------------

print("\nLoading datasets...")

X_train = pd.read_csv(os.path.join(INPUT_FOLDER, "X_train.csv"))
X_test = pd.read_csv(os.path.join(INPUT_FOLDER, "X_test.csv"))

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)
# ------------------------------------------------------------
# Fix invalid negative values
# ------------------------------------------------------------


# ------------------------------------------------------------
# Find Numerical Columns
# ------------------------------------------------------------

numeric_cols = X_train.select_dtypes(include=np.number).columns

# ------------------------------------------------------------
# Features NOT to transform
# ------------------------------------------------------------

exclude = [

    "Destination Port",

    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",

    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count"
]

# ------------------------------------------------------------
# Automatically detect skewed features
# ------------------------------------------------------------

skew = X_train[numeric_cols].skew()

log_features = []

for feature in skew.index:

    if feature in exclude:
        continue

    if abs(skew[feature]) > 1:

        # log1p only works safely when values >= 0
        if X_train[feature].min() >= 0:

            log_features.append(feature)

print("\nHighly Skewed Features Selected :", len(log_features))

for feature in log_features:
    print(f"{feature:<35} Skew = {skew[feature]:.2f}")
for feature in log_features:
    X_train[feature] = X_train[feature].clip(lower=0)
    X_test[feature] = X_test[feature].clip(lower=0)
# ------------------------------------------------------------
# Apply log1p
# ------------------------------------------------------------

print("\nApplying log(1+x)...")

X_train[log_features] = np.log1p(X_train[log_features])

X_test[log_features] = np.log1p(X_test[log_features])

# ------------------------------------------------------------
# Save transformed datasets
# ------------------------------------------------------------

X_train.to_csv(
    os.path.join(OUTPUT_FOLDER, "X_train_log.csv"),
    index=False
)

X_test.to_csv(
    os.path.join(OUTPUT_FOLDER, "X_test_log.csv"),
    index=False
)

joblib.dump(
    log_features,
    os.path.join(OUTPUT_FOLDER, "log_features.pkl")
)

print("\nSaved Files")

print("X_train_log.csv")
print("X_test_log.csv")
print("log_features.pkl")

print("\nSTEP 6 COMPLETED SUCCESSFULLY")    