import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

save_dir = r"D:\IDS research\feature_selection"

X_train = pd.read_csv(
    r"D:\IDS research\outputs\processed\X_train_balanced.csv"
)

model = joblib.load(
    r"D:\IDS research\cicids_project\models\lightgbm.pkl"
)

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    os.path.join(save_dir,
                 "feature_importance.csv"),
    index=False
)

top20 = importance.head(20)

print(top20)

plt.figure(figsize=(10,8))

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.title("Top 20 Important Features")

plt.tight_layout()

plt.savefig(
    os.path.join(save_dir,
                 "top20_features.png")
)

plt.close()

top20["Feature"].to_csv(
    os.path.join(save_dir,
                 "top20_names.csv"),
    index=False
)

print("Saved")