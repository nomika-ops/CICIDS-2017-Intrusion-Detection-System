import pandas as pd

X_train = pd.read_csv(r"D:\IDS research\outputs\X_train.csv")

features = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s"]

for col in features:
    neg = (X_train[col] < 0).sum()

    print(f"\n{col}")
    print(f"Negative Values : {neg}")
    print(f"Percentage      : {neg/len(X_train)*100:.6f}%")
    print(f"Minimum         : {X_train[col].min()}")