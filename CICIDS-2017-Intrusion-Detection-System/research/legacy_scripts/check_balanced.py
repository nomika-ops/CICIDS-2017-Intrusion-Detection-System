# check_balanced.py

import pandas as pd

y_original = pd.read_csv(
    r"D:\IDS research\outputs\y_train.csv"
)

y_balanced = pd.read_csv(
    r"D:\IDS research\outputs\processed\y_train_balanced.csv"
)

print("\nORIGINAL")
print(y_original.iloc[:,0].value_counts())

print("\nBALANCED")
print(y_balanced.iloc[:,0].value_counts())