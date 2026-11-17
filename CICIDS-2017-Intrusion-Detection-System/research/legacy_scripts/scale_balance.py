import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
base_dir = r"D:\IDS research\outputs"

X_train = pd.read_csv(os.path.join(base_dir, "X_train.csv"))
X_test = pd.read_csv(os.path.join(base_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(base_dir, "y_train.csv"))
y_test = pd.read_csv(os.path.join(base_dir, "y_test.csv"))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train.values.ravel())

os.makedirs(os.path.join(base_dir, "processed"), exist_ok=True)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
    os.path.join(base_dir, "processed", "X_train_scaled.csv"), index=False
)

pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
    os.path.join(base_dir, "processed", "X_test_scaled.csv"), index=False
)

pd.DataFrame(X_train_balanced, columns=X_train.columns).to_csv(
    os.path.join(base_dir, "processed", "X_train_balanced.csv"), index=False
)

pd.DataFrame(y_train_balanced, columns=["Label"]).to_csv(
    os.path.join(base_dir, "processed", "y_train_balanced.csv"), index=False
)

y_test.to_csv(os.path.join(base_dir, "processed", "y_test.csv"), index=False)

print("Scaling done.")
print("SMOTE balancing done.")
print("Processed files saved successfully.")
print("Balanced train label counts:")
print(pd.Series(y_train_balanced).value_counts())