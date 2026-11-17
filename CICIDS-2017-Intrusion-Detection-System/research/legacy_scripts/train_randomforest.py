# ============================================================
# RANDOM FOREST BINARY CLASSIFICATION
# ============================================================

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve
)

warnings.filterwarnings("ignore")

print("=" * 70)
print("RANDOM FOREST BINARY CLASSIFICATION")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

INPUT_FOLDER = r"D:\IDS research\outputs"
MODEL_FOLDER = r"D:\IDS research\models"
RESULT_FOLDER = r"D:\IDS research\results"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

X_train = pd.read_csv(
    os.path.join(INPUT_FOLDER, "X_train_scaled.csv")
)

X_test = pd.read_csv(
    os.path.join(INPUT_FOLDER, "X_test_scaled.csv")
)

y_train = pd.read_csv(
    os.path.join(INPUT_FOLDER, "y_train.csv")
).iloc[:, 0]

y_test = pd.read_csv(
    os.path.join(INPUT_FOLDER, "y_test.csv")
).iloc[:, 0]

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)
# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest...")

start_train = time.time()

rf = RandomForestClassifier(

    n_estimators=300,
    criterion="gini",

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    max_features="sqrt",

    bootstrap=True,

    random_state=42,

    n_jobs=-1

)

rf.fit(
    X_train,
    y_train
)

training_time = time.time() - start_train

print(f"Training Time : {training_time:.2f} sec")

# ============================================================
# PREDICTION
# ============================================================

print("\nPredicting...")

start_test = time.time()

y_pred = rf.predict(X_test)

y_prob = rf.predict_proba(X_test)[:, 1]

prediction_time = time.time() - start_test

print(f"Prediction Time : {prediction_time:.2f} sec")
# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating Model...")

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

pr_auc = average_precision_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

fpr = fp / (fp + tn)

fnr = fn / (fn + tp)

attack_miss = fn

print("\n==============================")
print("PRIMARY IDS METRICS")
print("==============================")

print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"PR AUC              : {pr_auc:.4f}")
print(f"Specificity         : {specificity:.4f}")
print(f"False Positive Rate : {fpr:.4f}")
print(f"False Negative Rate : {fnr:.4f}")

print("\nConfusion Matrix")
print(cm)

print(f"\nTP : {tp}")
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")

print(f"\nAttack Misses : {attack_miss}")

print(f"\nTraining Time : {training_time:.2f} sec")
print(f"Prediction Time : {prediction_time:.2f} sec")
# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\nPerforming Threshold Analysis...")

thresholds = np.arange(
    0.10,
    1.00,
    0.05
)

results = []

for threshold in thresholds:

    pred = (
        y_prob >= threshold
    ).astype(int)

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

        "False Negatives": fn_t,

        "False Negative Rate": fn_t / (fn_t + tp_t)

    })

threshold_df = pd.DataFrame(results)

threshold_df.to_csv(

    os.path.join(
        RESULT_FOLDER,
        "rf_threshold_analysis.csv"
    ),

    index=False

)

print("Threshold Analysis Saved.")
# ============================================================
# PRECISION RECALL CURVE
# ============================================================

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

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "rf_precision_recall_curve.png"

    ),

    dpi=300

)

plt.close()

# ============================================================
# RECALL VS F1
# ============================================================

plt.figure(figsize=(8,6))

plt.plot(

    threshold_df["Threshold"],

    threshold_df["Recall"],

    marker="o",

    label="Recall"

)

plt.plot(

    threshold_df["Threshold"],

    threshold_df["F1 Score"],

    marker="s",

    label="F1 Score"

)

plt.xlabel("Decision Threshold")

plt.ylabel("Score")

plt.title("Recall vs F1 Trade-off")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "rf_recall_f1_tradeoff.png"

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
    xticklabels=["Normal","Attack"],
    yticklabels=["Normal","Attack"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "rf_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()
# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nGenerating Feature Importance...")

importance = rf.feature_importances_

importance_df = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(

    os.path.join(
        RESULT_FOLDER,
        "rf_feature_importance.csv"
    ),

    index=False

)

plt.figure(figsize=(10,8))

plt.barh(

    importance_df["Feature"][:20],

    importance_df["Importance"][:20]

)

plt.gca().invert_yaxis()

plt.xlabel("Importance")

plt.title("Top 20 Important Features (Random Forest)")

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULT_FOLDER,
        "rf_feature_importance.png"
    ),

    dpi=300

)

plt.close()
# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Metric":[

        "Precision",

        "Recall",

        "F1 Score",

        "PR AUC",

        "Specificity",

        "False Positive Rate",

        "False Negative Rate",

        "False Negatives",

        "Training Time",

        "Prediction Time"

    ],

    "Value":[

        precision,

        recall,

        f1,

        pr_auc,

        specificity,

        fpr,

        fnr,

        attack_miss,

        training_time,

        prediction_time

    ]

})

metrics.to_csv(

    os.path.join(

        RESULT_FOLDER,

        "random_forest_metrics.csv"

    ),

    index=False

)
# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual":y_test,

    "Predicted":y_pred,

    "Probability":y_prob

})

prediction_df.to_csv(

    os.path.join(

        RESULT_FOLDER,

        "random_forest_predictions.csv"

    ),

    index=False

)
# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    digits=4
)

with open(

    os.path.join(

        RESULT_FOLDER,

        "random_forest_classification_report.txt"

    ),

    "w"

) as f:

    f.write(report)
# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n"+"="*70)

print("RANDOM FOREST COMPLETED SUCCESSFULLY")

print("="*70)

print(f"\nPrecision : {precision:.4f}")

print(f"Recall : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print(f"PR AUC : {pr_auc:.4f}")

print(f"Specificity : {specificity:.4f}")

print(f"False Positive Rate : {fpr:.4f}")

print(f"False Negative Rate : {fnr:.4f}")

print(f"Attack Misses : {attack_miss}")

print(f"Training Time : {training_time:.2f} sec")

print(f"Prediction Time : {prediction_time:.2f} sec")

print("\nGenerated Files")

print("- random_forest.pkl")

print("- random_forest_metrics.csv")

print("- random_forest_predictions.csv")

print("- random_forest_classification_report.txt")

print("- rf_feature_importance.csv")

print("- rf_feature_importance.png")

print("- rf_threshold_analysis.csv")

print("- rf_precision_recall_curve.png")

print("- rf_recall_f1_tradeoff.png")

print("- rf_confusion_matrix.png")

print("\nRandom Forest Completed Successfully.")