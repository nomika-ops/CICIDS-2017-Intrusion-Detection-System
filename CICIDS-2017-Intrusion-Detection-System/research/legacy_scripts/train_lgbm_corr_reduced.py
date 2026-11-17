import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

X_train = pd.read_csv(
    r"D:\IDS research\feature_selection\X_train_corr_reduced.csv"
)

X_test = pd.read_csv(
    r"D:\IDS research\feature_selection\X_test_corr_reduced.csv"
)

y_train = pd.read_csv(
    r"D:\IDS research\outputs\processed\y_train_balanced.csv"
)

y_test = pd.read_csv(
    r"D:\IDS research\outputs\processed\y_test.csv"
)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("Features Used:",
      X_train.shape[1])

model = LGBMClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nAccuracy:",
      accuracy_score(y_test,preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,preds))

print("\nClassification Report:")
print(classification_report(y_test,preds))