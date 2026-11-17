# ============================================================
# CATBOOST BINARY CLASSIFICATION
# ============================================================

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier

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

print("="*70)
print("CATBOOST BINARY CLASSIFICATION")
print("="*70)

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
).iloc[:,0]

y_test = pd.read_csv(
    os.path.join(INPUT_FOLDER, "y_test.csv")
).iloc[:,0]

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)
# ============================================================
# CATBOOST MODEL
# ============================================================

print("\nTraining CatBoost...")

start_train = time.time()

cb = CatBoostClassifier(

    iterations=300,

    learning_rate=0.05,

    depth=6,

    loss_function="Logloss",

    eval_metric="F1",

    random_seed=42,

    verbose=False,

    thread_count=-1

)

cb.fit(
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

y_pred = cb.predict(X_test)

y_pred = y_pred.astype(int)

y_prob = cb.predict_proba(X_test)[:,1]

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

accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n==============================")
print("PRIMARY IDS METRICS")
print("==============================")

print(f"Accuracy            : {accuracy:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"PR AUC              : {pr_auc:.4f}")
print(f"Specificity         : {specificity:.4f}")
print(f"False Positive Rate : {fpr:.4f}")
print(f"False Negative Rate : {fnr:.4f}")

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

    specificity_t = tn_t / (tn_t + fp_t)

    fpr_t = fp_t / (fp_t + tn_t)

    fnr_t = fn_t / (fn_t + tp_t)

    results.append({

        "Threshold": threshold,

        "Precision": precision_t,

        "Recall": recall_t,

        "F1 Score": f1_t,

        "Specificity": specificity_t,

        "False Positive Rate": fpr_t,

        "False Negative Rate": fnr_t,

        "False Negatives": fn_t

    })

threshold_df = pd.DataFrame(results)

threshold_df.to_csv(

    os.path.join(
        RESULT_FOLDER,
        "catboost_threshold_analysis.csv"
    ),

    index=False

)

print("Threshold Analysis Saved Successfully.")
# ============================================================
# PRECISION RECALL CURVE
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

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("CatBoost Precision-Recall Curve")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULT_FOLDER,
        "catboost_precision_recall_curve.png"
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

plt.xlabel("Decision Threshold")

plt.ylabel("Score")

plt.title("Recall vs F1 Trade-off (CatBoost)")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULT_FOLDER,
        "catboost_recall_f1_tradeoff.png"
    ),

    dpi=300

)

plt.close()
# ============================================================
# CONFUSION MATRIX HEATMAP
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

plt.title("CatBoost Confusion Matrix")

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULT_FOLDER,
        "catboost_confusion_matrix.png"
    ),

    dpi=300

)

plt.close()
# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nGenerating Feature Importance...")

importance = cb.get_feature_importance()

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

        "catboost_feature_importance.csv"

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

plt.title("Top 20 Important Features (CatBoost)")

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "catboost_feature_importance.png"

    ),

    dpi=300

)

plt.close()

# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

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

        accuracy,

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

        "catboost_metrics.csv"

    ),

    index=False

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

        "catboost_predictions.csv"

    ),

    index=False

)

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    cb,

    os.path.join(

        MODEL_FOLDER,

        "catboost_binary.pkl"

    )

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

        "catboost_classification_report.txt"

    ),

    "w"

) as f:

    f.write(report)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)

print("CATBOOST BINARY CLASSIFICATION COMPLETED")

print("="*70)

print(f"\nAccuracy            : {accuracy:.4f}")

print(f"Precision           : {precision:.4f}")

print(f"Recall              : {recall:.4f}")

print(f"F1 Score            : {f1:.4f}")

print(f"PR AUC              : {pr_auc:.4f}")

print(f"Specificity         : {specificity:.4f}")

print(f"False Positive Rate : {fpr:.4f}")

print(f"False Negative Rate : {fnr:.4f}")

print(f"Attack Misses (FN)  : {attack_miss}")

print(f"\nTraining Time       : {training_time:.2f} sec")

print(f"Prediction Time     : {prediction_time:.2f} sec")

print("\nGenerated Files:")

print("- catboost_binary.pkl")

print("- catboost_metrics.csv")

print("- catboost_predictions.csv")

print("- catboost_classification_report.txt")

print("- catboost_feature_importance.csv")

print("- catboost_feature_importance.png")

print("- catboost_threshold_analysis.csv")

print("- catboost_precision_recall_curve.png")

print("- catboost_recall_f1_tradeoff.png")

print("- catboost_confusion_matrix.png")

print("\nProject Completed Successfully.")