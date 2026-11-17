import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_FOLDER = r"D:\IDS research\results"

print("="*70)
print("MODEL COMPARISON")
print("="*70)

models = {
    "Random Forest":"random_forest_metrics.csv",
    "XGBoost":"xgboost_metrics.csv",
    "LightGBM":"lightgbm_metrics.csv",
    "Extra Trees":"extra_trees_metrics.csv",
    "Decision Tree":"decision_tree_metrics.csv",
    "Logistic Regression":"logistic_regression_metrics.csv",
    "Naive Bayes":"naive_bayes_metrics.csv"
}

comparison = []

for model,file in models.items():

    path = os.path.join(RESULT_FOLDER,file)

    if os.path.exists(path):

        df = pd.read_csv(path)

        metrics = dict(zip(df["Metric"],df["Value"]))

        metrics["Model"] = model

        comparison.append(metrics)

comparison = pd.DataFrame(comparison)

comparison = comparison[[
    "Model",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "ROC_AUC",
    "MCC",
    "Training_Time",
    "Prediction_Time"
]]

comparison = comparison.sort_values(
    by="Macro F1",
    ascending=False
)

comparison.reset_index(drop=True,inplace=True)
comparison.to_csv(

    os.path.join(
        RESULT_FOLDER,
        "model_comparison.csv"
    ),

    index=False

)

print("\nComparison Table")

print(comparison)

print("\nSaved:")
print("model_comparison.csv")
plots = [
    ("Accuracy","accuracy_comparison.png"),
    ("Macro F1","macro_f1_comparison.png"),
    ("MCC","mcc_comparison.png"),
    ("ROC_AUC","roc_auc_comparison.png"),
    ("Training_Time","training_time_comparison.png")
]

for metric,filename in plots:

    plt.figure(figsize=(10,6))

    plt.bar(
        comparison["Model"],
        comparison[metric]
    )

    plt.xticks(rotation=30,ha="right")

    plt.ylabel(metric)

    plt.title(metric + " Comparison")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            RESULT_FOLDER,
            filename
        ),

        dpi=300

    )

    plt.close()

print("Graphs Generated Successfully.")
print("\n"+"="*70)

print("FINAL MODEL RANKING")

print("="*70)

for i,row in comparison.iterrows():

    print(
        f"{i+1}. {row['Model']}"
        f" | Accuracy={row['Accuracy']:.4f}"
        f" | Macro F1={row['Macro F1']:.4f}"
        f" | MCC={row['MCC']:.4f}"
    )

best = comparison.iloc[0]

print("\nBest Performing Model")

print("----------------------------")

print("Model        :",best["Model"])
print("Accuracy     :",round(best["Accuracy"],4))
print("Macro F1     :",round(best["Macro F1"],4))
print("ROC_AUC      :",round(best["ROC_AUC"],4))
print("MCC          :",round(best["MCC"],4))

print("\nComparison Completed Successfully.")

print("="*70)