import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data_dir = r"D:\IDS research\outputs\processed"
model_dir = r"D:\IDS research\cicids_project\models"

X_train = pd.read_csv(os.path.join(data_dir, "X_train_balanced.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train_balanced.csv"))
X_test = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

if X_train.shape[0] != len(y_train):
    raise ValueError("X_train and y_train row counts do not match.")

if X_test.shape[0] != len(y_test):
    raise ValueError("X_test and y_test row counts do not match.")

print("\nTraining balanced model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nBalanced Model Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "balanced_rf.pkl"))

print("\nBalanced model saved successfully.")