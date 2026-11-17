import pandas as pd

features = pd.read_csv(
    r"D:\IDS research\feature_selection\top20_names.csv",
    header=None
)[0].tolist()

X_train = pd.read_csv(
    r"D:\IDS research\outputs\processed\X_train_balanced.csv"
)

X_test = pd.read_csv(
    r"D:\IDS research\outputs\processed\X_test_scaled.csv"
)

X_train_top20 = X_train[features]
X_test_top20 = X_test[features]

X_train_top20.to_csv(
    r"D:\IDS research\feature_selection\X_train_top20.csv",
    index=False
)

X_test_top20.to_csv(
    r"D:\IDS research\feature_selection\X_test_top20.csv",
    index=False
)

print("Top20 datasets saved.")