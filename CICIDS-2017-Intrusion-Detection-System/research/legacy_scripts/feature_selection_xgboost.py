import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ============================================================
# FEATURE SELECTION USING XGBOOST FEATURE IMPORTANCE
# ============================================================

print("=" * 70)
print("XGBOOST FEATURE SELECTION EXPERIMENTS")
print("=" * 70)

INPUT_FOLDER = r"D:\IDS research\outputs"
RESULT_FOLDER = r"D:\IDS research\results"
MODEL_FOLDER = r"D:\IDS research\models"

os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

print("\nLoading datasets...")

X_train = pd.read_csv(
    os.path.join(INPUT_FOLDER, "X_train_balanced.csv")
)

X_test = pd.read_csv(
    os.path.join(INPUT_FOLDER, "X_test_scaled.csv")
)

y_train = pd.read_csv(
    os.path.join(INPUT_FOLDER, "y_train_balanced.csv")
).iloc[:, 0]

y_test = pd.read_csv(
    os.path.join(INPUT_FOLDER, "y_test.csv")
).iloc[:, 0]

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ------------------------------------------------------------
# Load Feature Importance
# ------------------------------------------------------------

importance = pd.read_csv(
    os.path.join(
        RESULT_FOLDER,
        "xgb_feature_importance.csv"
    )
)

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

feature_sets = [50, 40, 30, 20, 10]

results = []

classes = np.unique(y_test)

y_test_bin = label_binarize(
    y_test,
    classes=classes
)
# ============================================================
# TRAINING LOOP
# ============================================================

for top_n in feature_sets:

    print("\n" + "=" * 60)
    print(f"TOP {top_n} FEATURES")
    print("=" * 60)

    selected_features = importance.head(top_n)["Feature"].tolist()

    X_train_fs = X_train[selected_features]
    X_test_fs = X_test[selected_features]

    print("Training Shape :", X_train_fs.shape)
    print("Testing Shape  :", X_test_fs.shape)

    # --------------------------------------------------------
    # XGBoost Model
    # --------------------------------------------------------

    xgb = XGBClassifier(

        objective="multi:softprob",

        num_class=len(np.unique(y_train)),

        n_estimators=300,

        learning_rate=0.1,

        max_depth=8,

        min_child_weight=3,

        subsample=0.8,

        colsample_bytree=0.8,

        gamma=0.1,

        reg_alpha=0,

        reg_lambda=1,

        random_state=42,

        n_jobs=-1,

        tree_method="hist",

        eval_metric="mlogloss"

    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("Training Model...")

    start_train = time.time()

    xgb.fit(
        X_train_fs,
        y_train
    )

    training_time = time.time() - start_train

    print(
        "Training Time :",
        round(training_time, 2),
        "seconds"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("Predicting...")

    start_test = time.time()

    y_pred = xgb.predict(
        X_test_fs
    )

    y_prob = xgb.predict_proba(
        X_test_fs
    )

    prediction_time = time.time() - start_test

    print(
        "Prediction Time :",
        round(prediction_time, 2),
        "seconds"
    )

    # --------------------------------------------------------
    # Primary Metrics
    # --------------------------------------------------------

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

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test_bin,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

    print("\nPRIMARY METRICS")
    print("-" * 40)
    print("Macro F1        :", round(macro_f1, 4))
    print("MCC             :", round(mcc, 4))
    print("Macro Recall    :", round(macro_recall, 4))
    print("Macro Precision :", round(macro_precision, 4))
    print("ROC-AUC         :", round(roc_auc, 4))
    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    model_name = f"xgboost_top_{top_n}.pkl"

    joblib.dump(

        xgb,

        os.path.join(

            MODEL_FOLDER,

            model_name

        )

    )

    # --------------------------------------------------------
    # Save Individual Metrics
    # --------------------------------------------------------

    metrics = pd.DataFrame({

        "Metric": [

            "Macro F1",

            "MCC",

            "Macro Recall",

            "Macro Precision",

            "ROC AUC",

            "Training Time",

            "Prediction Time"

        ],

        "Value": [

            macro_f1,

            mcc,

            macro_recall,

            macro_precision,

            roc_auc,

            training_time,

            prediction_time

        ]

    })

    metrics.to_csv(

        os.path.join(

            RESULT_FOLDER,

            f"top_{top_n}_metrics.csv"

        ),

        index=False

    )

    # --------------------------------------------------------
    # Store Results
    # --------------------------------------------------------

    results.append({

        "Features": top_n,

        "Macro F1": macro_f1,

        "MCC": mcc,

        "Macro Recall": macro_recall,

        "Macro Precision": macro_precision,

        "ROC AUC": roc_auc,

        "Training Time": training_time,

        "Prediction Time": prediction_time

    })

    print("\nModel Saved :", model_name)

    print("Metrics Saved :", f"top_{top_n}_metrics.csv")
# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Features",
    ascending=False
)

results_df.to_csv(

    os.path.join(

        RESULT_FOLDER,

        "feature_selection_results.csv"

    ),

    index=False

)

print("\nCombined Results Saved Successfully.")

# ============================================================
# PLOT 1 : Macro F1 vs Number of Features
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(

    results_df["Features"],

    results_df["Macro F1"],

    marker="o",

    linewidth=2

)

plt.gca().invert_xaxis()

plt.grid(True)

plt.xlabel("Number of Selected Features")

plt.ylabel("Macro F1")

plt.title("Macro F1 vs Number of Features")

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "macro_f1_vs_features.png"

    ),

    dpi=300

)

plt.close()

# ============================================================
# PLOT 2 : Training Time vs Number of Features
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(

    results_df["Features"],

    results_df["Training Time"],

    marker="o",

    linewidth=2

)

plt.gca().invert_xaxis()

plt.grid(True)

plt.xlabel("Number of Selected Features")

plt.ylabel("Training Time (sec)")

plt.title("Training Time vs Number of Features")

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "training_time_vs_features.png"

    ),

    dpi=300

)

plt.close()

# ============================================================
# PLOT 3 : Prediction Time vs Number of Features
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(

    results_df["Features"],

    results_df["Prediction Time"],

    marker="o",

    linewidth=2

)

plt.gca().invert_xaxis()

plt.grid(True)

plt.xlabel("Number of Selected Features")

plt.ylabel("Prediction Time (sec)")

plt.title("Prediction Time vs Number of Features")

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_FOLDER,

        "prediction_time_vs_features.png"

    ),

    dpi=300

)

plt.close()

print("Performance graphs generated successfully.")
# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST FEATURE SELECTION RESULTS")
print("=" * 70)

print("\nSummary Table\n")

summary = results_df.copy()

summary["Macro F1"] = summary["Macro F1"].round(4)
summary["MCC"] = summary["MCC"].round(4)
summary["Macro Recall"] = summary["Macro Recall"].round(4)
summary["Macro Precision"] = summary["Macro Precision"].round(4)
summary["ROC AUC"] = summary["ROC AUC"].round(4)
summary["Training Time"] = summary["Training Time"].round(2)
summary["Prediction Time"] = summary["Prediction Time"].round(2)

print(summary.to_string(index=False))

# ============================================================
# BEST FEATURE SET
# ============================================================

best = results_df.loc[
    results_df["Macro F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST FEATURE SUBSET")
print("=" * 70)

print("Selected Features :", int(best["Features"]))
print("Macro F1          :", round(best["Macro F1"], 4))
print("MCC               :", round(best["MCC"], 4))
print("Macro Recall      :", round(best["Macro Recall"], 4))
print("Macro Precision   :", round(best["Macro Precision"], 4))
print("ROC AUC           :", round(best["ROC AUC"], 4))
print("Training Time     :", round(best["Training Time"], 2), "sec")
print("Prediction Time   :", round(best["Prediction Time"], 2), "sec")

# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    os.path.join(
        RESULT_FOLDER,
        "feature_selection_summary.csv"
    ),
    index=False
)

# ============================================================
# OUTPUT FILES
# ============================================================

print("\nGenerated Files")
print("- feature_selection_results.csv")
print("- feature_selection_summary.csv")
print("- macro_f1_vs_features.png")
print("- training_time_vs_features.png")
print("- prediction_time_vs_features.png")

for n in feature_sets:
    print(f"- top_{n}_metrics.csv")
    print(f"- xgboost_top_{n}.pkl")

print("=" * 70)
print("FEATURE SELECTION EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)