# ============================================================
# XGBOOST MULTICLASS CLASSIFICATION
# ============================================================

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

print("="*70)
print("XGBOOST MULTICLASS CLASSIFICATION")
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
    os.path.join(
        INPUT_FOLDER,
        "X_train_multi_class_scaled.csv"
    )
)

X_test = pd.read_csv(
    os.path.join(
        INPUT_FOLDER,
        "X_test_multi_class_scaled.csv"
    )
)

y_train = pd.read_csv(
    os.path.join(
        INPUT_FOLDER,
        "y_train_multi_class.csv"
    )
).iloc[:,0]

y_test = pd.read_csv(
    os.path.join(
        INPUT_FOLDER,
        "y_test_multi_class.csv"
    )
).iloc[:,0]

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

num_classes = len(np.unique(y_train))

print("Number of Classes :", num_classes)
# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\nTraining XGBoost...")

start_train = time.time()

xgb = XGBClassifier(

    objective="multi:softmax",

    num_class=num_classes,

    n_estimators=300,

    max_depth=8,

    learning_rate=0.1,

    subsample=0.8,

    colsample_bytree=0.8,

    eval_metric="mlogloss",

    random_state=42,

    n_jobs=-1

)

xgb.fit(

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

y_pred = xgb.predict(X_test)

prediction_time = time.time() - start_test

print(f"Prediction Time : {prediction_time:.2f} sec")

# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating Model...")

accuracy = accuracy_score(
    y_test,
    y_pred
)

weighted_precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred,
    digits=4,
    zero_division=0
)

print("\n==============================")
print("PRIMARY IDS METRICS")
print("==============================")

print(f"Accuracy              : {accuracy:.4f}")
print(f"Weighted Precision    : {weighted_precision:.4f}")
print(f"Weighted Recall       : {weighted_recall:.4f}")
print(f"Weighted F1 Score     : {weighted_f1:.4f}")

print()

print(f"Macro Precision       : {macro_precision:.4f}")
print(f"Macro Recall          : {macro_recall:.4f}")
print(f"Macro F1 Score        : {macro_f1:.4f}")

print()

print(f"Training Time         : {training_time:.2f} sec")
print(f"Prediction Time       : {prediction_time:.2f} sec")

print("\nClassification Report")
print("="*70)

print(report)
# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(

    os.path.join(
        RESULT_FOLDER,
        "xgboost_multiclass_classification_report.txt"
    ),

    "w"

) as f:

    f.write(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating Confusion Matrix...")

plt.figure(figsize=(12,10))

sns.heatmap(

    cm,

    cmap="Blues",

    fmt="d"

)

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.title("XGBoost Multiclass Confusion Matrix")

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULT_FOLDER,
        "xgboost_multiclass_confusion_matrix.png"
    ),

    dpi=300

)

plt.close()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nGenerating Feature Importance...")

importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": xgb.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

importance.to_csv(

    os.path.join(

        RESULT_FOLDER,

        "xgboost_multiclass_feature_importance.csv"

    ),

    index=False

)

plt.figure(figsize=(10,8))

plt.barh(

    importance["Feature"][:20],

    importance["Importance"][:20]

)

plt.gca().invert_yaxis()

plt.xlabel("Importance")

plt.title("Top 20 Important Features")

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "xgboost_multiclass_feature_importance.png"

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

        "Weighted Precision",

        "Weighted Recall",

        "Weighted F1",

        "Macro Precision",

        "Macro Recall",

        "Macro F1",

        "Training Time",

        "Prediction Time"

    ],

    "Value":[

        accuracy,

        weighted_precision,

        weighted_recall,

        weighted_f1,

        macro_precision,

        macro_recall,

        macro_f1,

        training_time,

        prediction_time

    ]

})

metrics.to_csv(

    os.path.join(

        RESULT_FOLDER,

        "xgboost_multiclass_metrics.csv"

    ),

    index=False

)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_pred

})

prediction_df.to_csv(

    os.path.join(

        RESULT_FOLDER,

        "xgboost_multiclass_predictions.csv"

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

        "xgboost_multiclass.pkl"

    )

)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)

print("XGBOOST MULTICLASS COMPLETED")

print("="*70)

print(f"Accuracy              : {accuracy:.4f}")
print(f"Weighted Precision    : {weighted_precision:.4f}")
print(f"Weighted Recall       : {weighted_recall:.4f}")
print(f"Weighted F1 Score     : {weighted_f1:.4f}")

print()

print(f"Macro Precision       : {macro_precision:.4f}")
print(f"Macro Recall          : {macro_recall:.4f}")
print(f"Macro F1 Score        : {macro_f1:.4f}")

print()

print(f"Training Time         : {training_time:.2f} sec")
print(f"Prediction Time       : {prediction_time:.2f} sec")

print("\nGenerated Files")

print("- xgboost_multiclass.pkl")
print("- xgboost_multiclass_metrics.csv")
print("- xgboost_multiclass_predictions.csv")
print("- xgboost_multiclass_classification_report.txt")
print("- xgboost_multiclass_feature_importance.csv")
print("- xgboost_multiclass_feature_importance.png")
print("- xgboost_multiclass_confusion_matrix.png")

print("\nXGBoost Multiclass Completed Successfully.")