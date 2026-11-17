import pandas as pd
import os

from imblearn.over_sampling import BorderlineSMOTE

print("=" * 70)
print("STEP 5B : BORDERLINE SMOTE")
print("=" * 70)

INPUT_FOLDER = r"D:\IDS research\outputs"
OUTPUT_FOLDER = r"D:\IDS research\outputs"

print("\nLoading under-sampled dataset...")

X_train = pd.read_csv(
    os.path.join(INPUT_FOLDER, "X_train_under.csv")
)

y_train = pd.read_csv(
    os.path.join(INPUT_FOLDER, "y_train_under.csv")
)

if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]

print("Training Shape :", X_train.shape)

print("\nCurrent Distribution\n")
print(y_train.value_counts().sort_index())

sampling_strategy = {

    1:15000,
    3:15000,
    5:15000,
    6:15000,
    7:15000,
    11:15000,
    12:8000,
    14:5000

}

smote = BorderlineSMOTE(

    sampling_strategy=sampling_strategy,

    random_state=42,

    k_neighbors=3,

    kind="borderline-1"

)

print("\nApplying BorderlineSMOTE...")

X_balanced, y_balanced = smote.fit_resample(

    X_train,

    y_train

)
print("\nBalanced Distribution\n")
print(pd.Series(y_balanced).value_counts().sort_index())

pd.DataFrame(

    X_balanced,

    columns=X_train.columns

).to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "X_train_balanced.csv"
    ),

    index=False

)

pd.DataFrame(

    y_balanced,

    columns=["Label"]

).to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "y_train_balanced.csv"
    ),

    index=False

)

print("\nBalanced Shape :", X_balanced.shape)

print("\nSaved Files")

print("X_train_balanced.csv")
print("y_train_balanced.csv")

print("\nSTEP 5B COMPLETED SUCCESSFULLY")
print("=" * 70)