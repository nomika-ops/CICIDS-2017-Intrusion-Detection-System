# ============================================================
# RANDOM FOREST MULTICLASS CLASSIFICATION
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

print("="*70)
print("RANDOM FOREST MULTICLASS CLASSIFICATION")
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
print("Number of Classes :", len(np.unique(y_train)))
# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

start_train = time.time()

rf = RandomForestClassifier(

    n_estimators=300,

    max_depth=30,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    n_jobs=-1,

    random_state=42

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

# Weighted Metrics
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

# Macro Metrics (important for minority attacks)

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

# Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

# Classification Report

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