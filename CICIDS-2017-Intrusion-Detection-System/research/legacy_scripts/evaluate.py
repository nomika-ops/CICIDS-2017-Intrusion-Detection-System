import pandas as pd
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

data_dir = r"D:\IDS research\outputs\processed"
model_dir = r"D:\IDS research\cicids_project\models" 

# Load test data
X_test = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
y_test = y_test.values.ravel()

print("Loading both models...")

# Load baseline model
baseline_model = joblib.load(os.path.join(model_dir, "baseline_rf.pkl"))
y_pred_baseline = baseline_model.predict(X_test)

# Load balanced model
balanced_model = joblib.load(os.path.join(model_dir, "balanced_rf.pkl"))
y_pred_balanced = balanced_model.predict(X_test)

print("\n" + "="*70)
print("BASELINE (IMBALANCED) MODEL RESULTS")
print("="*70)
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_baseline))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline))

print("\n" + "="*70)
print("BALANCED MODEL RESULTS")
print("="*70)
print("Accuracy:", accuracy_score(y_test, y_pred_balanced))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_balanced))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_balanced))

# Save comparison to JSON
results = {
    "baseline": {
        "accuracy": accuracy_score(y_test, y_pred_baseline),
        "confusion_matrix": confusion_matrix(y_test, y_pred_baseline).tolist(),
        "classification_report": {k: v for k, v in classification_report(y_test, y_pred_baseline, output_dict=True).items()}
    },
    "balanced": {
        "accuracy": accuracy_score(y_test, y_pred_balanced),
        "confusion_matrix": confusion_matrix(y_test, y_pred_balanced).tolist(),
        "classification_report": {k: v for k, v in classification_report(y_test, y_pred_balanced, output_dict=True).items()}
    }
}

os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, "comparison_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"Balanced Accuracy: {accuracy_score(y_test, y_pred_balanced):.4f}")
print(f"Difference: {accuracy_score(y_test, y_pred_balanced) - accuracy_score(y_test, y_pred_baseline):.4f}")
print("\nComparison saved to: comparison_results.json")