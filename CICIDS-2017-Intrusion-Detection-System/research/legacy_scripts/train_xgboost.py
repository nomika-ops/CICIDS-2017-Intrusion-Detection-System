import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve
)
# ============================================================
# XGBOOST BINARY CLASSIFICATION
# ============================================================

print("=" * 70)
print("XGBOOST BINARY CLASSIFICATION")
print("=" * 70)
INPUT_FOLDER = r"D:\IDS research\outputs"
MODEL_FOLDER = r"D:\IDS research\models"
RESULT_FOLDER = r"D:\IDS research\results"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

print("\nLoading datasets...")

X_train = pd.read_csv(os.path.join(INPUT_FOLDER, "X_train_scaled.csv"))
X_test = pd.read_csv(os.path.join(INPUT_FOLDER, "X_test_scaled.csv"))

y_train = pd.read_csv(
    os.path.join(INPUT_FOLDER, "y_train.csv")
).iloc[:, 0]

y_test = pd.read_csv(
    os.path.join(INPUT_FOLDER, "y_test.csv")
).iloc[:, 0]

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)
print("\nTraining XGBoost...")

start_train = time.time()

xgb = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

training_time = time.time() - start_train

print(f"Training Time : {training_time:.2f} sec")
print("\nPredicting...")

start_test = time.time()

y_pred = xgb.predict(X_test)

y_prob = xgb.predict_proba(X_test)[:, 1]

prediction_time = time.time() - start_test

print(f"Prediction Time : {prediction_time:.2f} sec")
# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating Model...")

# Primary Metrics
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

# Precision-Recall AUC
pr_auc = average_precision_score(
    y_test,
    y_prob
)

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

# Attack Misses
attack_miss = fn

print("\n==============================")
print("PRIMARY IDS METRICS")
print("==============================")

print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"PR AUC              : {pr_auc:.4f}")

print("\nConfusion Matrix")
print(cm)

print(f"\nTrue Positives      : {tp}")
print(f"True Negatives      : {tn}")
print(f"False Positives     : {fp}")
print(f"False Negatives     : {fn}")

print(f"\nAttack Misses (FN)  : {attack_miss}")

print(f"\nTraining Time       : {training_time:.2f} sec")
print(f"Prediction Time     : {prediction_time:.2f} sec")
# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\nPerforming Threshold Analysis...")

thresholds = np.arange(0.10, 1.00, 0.05)

results = []

for threshold in thresholds:

    pred = (y_prob >= threshold).astype(int)

    precision_t = precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall_t = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1_t = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    tn_t, fp_t, fn_t, tp_t = confusion_matrix(
        y_test,
        pred
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision_t,
        "Recall": recall_t,
        "F1 Score": f1_t,
        "False Negatives": fn_t
    })

threshold_df = pd.DataFrame(results)

threshold_df.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "threshold_analysis.csv"
    ),
    index=False
)

print("Threshold analysis saved successfully.")
# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

print("\nGenerating Precision-Recall Curve...")

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    recall_curve,
    precision_curve,
    linewidth=2,
    label=f"PR AUC = {pr_auc:.4f}"
)

plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision-Recall Curve", fontsize=14)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "precision_recall_curve.png"
    ),
    dpi=300
)

plt.close()
# ============================================================
# RECALL vs F1 TRADE-OFF
# ============================================================

print("\nGenerating Recall vs F1 Trade-off Graph...")

plt.figure(figsize=(8,6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    linewidth=2,
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1 Score"],
    marker="s",
    linewidth=2,
    label="F1 Score"
)

plt.xlabel("Decision Threshold", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Recall vs F1 Score", fontsize=14)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "recall_f1_tradeoff.png"
    ),
    dpi=300
)

plt.close()
# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating Confusion Matrix...")

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()
# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1 Score",
        "PR AUC",
        "False Negatives",
        "Training Time (sec)",
        "Prediction Time (sec)"
    ],
    "Value": [
        precision,
        recall,
        f1,
        pr_auc,
        attack_miss,
        training_time,
        prediction_time
    ]
})

metrics.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "xgboost_binary_metrics.csv"
    ),
    index=False
)
# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    xgb,
    os.path.join(
        MODEL_FOLDER,
        "xgboost_binary.pkl"
    )
)
# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Probability": y_prob
})

prediction_df.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "xgboost_binary_predictions.csv"
    ),
    index=False
)
# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST BINARY CLASSIFICATION COMPLETED")
print("=" * 70)

print(f"\nPrecision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"PR AUC             : {pr_auc:.4f}")
print(f"Attack Misses (FN) : {attack_miss}")

print("\nGenerated Files:")

print("- xgboost_binary.pkl")
print("- xgboost_binary_metrics.csv")
print("- xgboost_binary_predictions.csv")
print("- threshold_analysis.csv")
print("- precision_recall_curve.png")
print("- recall_f1_tradeoff.png")
print("- confusion_matrix.png")
print("- feature_importance.png")

print("\nProject Completed Successfully.")