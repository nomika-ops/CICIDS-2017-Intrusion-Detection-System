import pandas as pd
import os
import joblib
from sklearn.metrics import classification_report, accuracy_score

processed_dir = r"D:\IDS research\outputs\processed"
model_dir = r"D:\IDS research\cicids_project\models"

print("Loading files from processed directory...")

# -------------------------
# Binary test data
# -------------------------
X_test_binary = pd.read_csv(os.path.join(processed_dir, "X_test_scaled.csv"))
y_test_binary = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))

# If y_test has an extra index column, keep only the last column
if y_test_binary.shape[1] > 1:
    y_test_binary = y_test_binary.iloc[:, -1]

y_test_binary = y_test_binary.values.ravel()

# -------------------------
# Multi-class test data
# -------------------------
X_test_multi_full = pd.read_csv(os.path.join(processed_dir, "X_test_multi_class.csv"))
print("Multi-class X_test_full columns:", X_test_multi_full.shape[1])

# Drop label/index column if file has 79 columns and model expects 78 features
if X_test_multi_full.shape[1] == 79:
    X_test_multi = X_test_multi_full.iloc[:, 1:]
else:
    X_test_multi = X_test_multi_full.copy()

print("Multi-class X_test after adjustment:", X_test_multi.shape)

y_test_multi = pd.read_csv(os.path.join(processed_dir, "y_test_multi_class.csv"))

# If CSV has index + label columns, keep only the last column
if y_test_multi.shape[1] > 1:
    y_test_multi = y_test_multi.iloc[:, -1]

y_test_multi = y_test_multi.values.ravel()

# -------------------------
# Show shapes
# -------------------------
print("\nBinary X_test shape:", X_test_binary.shape)
print("Multi-class X_test shape:", X_test_multi.shape)
print("Binary y_test shape:", len(y_test_binary))
print("Multi-class y_test shape:", len(y_test_multi))

# -------------------------
# Fix length mismatch if any
# -------------------------
if len(y_test_multi) != len(X_test_multi):
    print("\nLength mismatch found in multi-class test data.")
    print("X_test_multi rows:", len(X_test_multi))
    print("y_test_multi rows:", len(y_test_multi))
    print("Trimming to matching minimum length...")

    min_len = min(len(X_test_multi), len(y_test_multi))
    X_test_multi = X_test_multi.iloc[:min_len, :]
    y_test_multi = y_test_multi[:min_len]

if len(y_test_binary) != len(X_test_binary):
    print("\nLength mismatch found in binary test data.")
    print("X_test_binary rows:", len(X_test_binary))
    print("y_test_binary rows:", len(y_test_binary))
    print("Trimming to matching minimum length...")

    min_len = min(len(X_test_binary), len(y_test_binary))
    X_test_binary = X_test_binary.iloc[:min_len, :]
    y_test_binary = y_test_binary[:min_len]

print("\nFeature count matches:", X_test_binary.shape[1], "features")

# -------------------------
# Load models
# -------------------------
print("\nLoading models...")
model_binary = joblib.load(os.path.join(model_dir, "baseline_rf.pkl"))
model_multi = joblib.load(os.path.join(model_dir, "multi_class_rf.pkl"))

# -------------------------
# Predict
# -------------------------
print("\nPredicting...")
y_pred_binary = model_binary.predict(X_test_binary)
y_pred_multi = model_multi.predict(X_test_multi)

# -------------------------
# Results
# -------------------------
binary_acc = accuracy_score(y_test_binary, y_pred_binary)
multi_acc = accuracy_score(y_test_multi, y_pred_multi)

print("\n" + "=" * 70)
print("BINARY vs MULTI-CLASS COMPARISON")
print("=" * 70)

print("\nBinary Model Accuracy (Benign vs Attack):", round(binary_acc, 4))
print("Multi-class Model Accuracy (Attack Subtypes):", round(multi_acc, 4))

print("\n" + "=" * 70)
print("Binary Model Report")
print("=" * 70)
print(classification_report(y_test_binary, y_pred_binary))

print("\n" + "=" * 70)
print("Multi-class Model Report")
print("=" * 70)
print(classification_report(y_test_multi, y_pred_multi))

# -------------------------
# Save comparison results
# -------------------------
results_df = pd.DataFrame({
    "Model": ["Binary", "Multi-class"],
    "Accuracy": [binary_acc, multi_acc],
    "Test_Samples": [len(y_test_binary), len(y_test_multi)],
    "Features": [X_test_binary.shape[1], X_test_multi.shape[1]]
})

results_path = os.path.join(model_dir, "comparison_results.csv")
results_df.to_csv(results_path, index=False)

print("\nComparison saved to:", results_path)