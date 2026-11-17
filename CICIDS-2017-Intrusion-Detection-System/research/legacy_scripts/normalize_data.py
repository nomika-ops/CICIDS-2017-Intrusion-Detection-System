import pandas as pd
import joblib
import os

from sklearn.preprocessing import RobustScaler

# ============================================================
#  NORMALIZATION
# ============================================================

print("=" * 70)
print("Starting NORMALIZATION")
print("=" * 70)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

INPUT_FOLDER = r"D:\IDS research\outputs"
OUTPUT_FOLDER = r"D:\IDS research\outputs"

# ------------------------------------------------------------
# Load Train-Test Data
# ------------------------------------------------------------

print("\nLoading datasets...")

X_train = pd.read_csv(os.path.join(INPUT_FOLDER, "X_train_log.csv"))
X_test = pd.read_csv(os.path.join(INPUT_FOLDER, "X_test_log.csv"))

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ------------------------------------------------------------
# Initialize RobustScaler
# ------------------------------------------------------------

print("\nApplying RobustScaler...")

scaler = RobustScaler()

# ------------------------------------------------------------
# Fit ONLY on Training Data
# ------------------------------------------------------------

X_train_scaled = scaler.fit_transform(X_train)

# ------------------------------------------------------------
# Transform Test Data
# ------------------------------------------------------------

X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# Convert Back to DataFrame
# ------------------------------------------------------------

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

# ------------------------------------------------------------
# Save Normalized Data
# ------------------------------------------------------------

X_train_scaled.to_csv(
    os.path.join(OUTPUT_FOLDER, "X_train_scaled.csv"),
    index=False
)

X_test_scaled.to_csv(
    os.path.join(OUTPUT_FOLDER, "X_test_scaled.csv"),
    index=False
)

# ------------------------------------------------------------
# Save Scaler
# ------------------------------------------------------------

joblib.dump(
    scaler,
    os.path.join(OUTPUT_FOLDER, "robust_scaler.pkl")
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nNormalization Completed Successfully.")

print("\nSaved Files:")

print("X_train_scaled.csv")
print("X_test_scaled.csv")
print("robust_scaler.pkl")

print("\nTraining Shape :", X_train_scaled.shape)
print("Testing Shape  :", X_test_scaled.shape)

print("\nNormalization COMPLETED SUCCESSFULLY")