import pandas as pd
import os

from imblearn.under_sampling import RandomUnderSampler

print("="*70)
print("STEP 5A : RANDOM UNDER SAMPLING")
print("="*70)

INPUT_FOLDER = r"D:\IDS research\outputs"
OUTPUT_FOLDER = r"D:\IDS research\outputs"

print("\nLoading datasets...")

X_train = pd.read_csv(
    os.path.join(INPUT_FOLDER,"X_train_scaled.csv")
)

y_train = pd.read_csv(
    os.path.join(INPUT_FOLDER,"y_train.csv")
)

if isinstance(y_train,pd.DataFrame):
    y_train = y_train.iloc[:,0]

print("Training Shape :",X_train.shape)

print("\nOriginal Distribution\n")
print(y_train.value_counts().sort_index())

sampling_strategy={

0:250000,

1:1558,

2:102411,

3:8229,

4:138277,

5:4182,

6:4308,

7:4745,

8:9,

9:29,

10:72555,

11:2575,

12:1176,

13:17,

14:522

}

rus=RandomUnderSampler(

sampling_strategy=sampling_strategy,

random_state=42

)

X_under,y_under=rus.fit_resample(

X_train,

y_train

)
print("\nAfter Under Sampling\n")

print(pd.Series(y_under).value_counts().sort_index())

pd.DataFrame(

X_under,

columns=X_train.columns

).to_csv(

os.path.join(

OUTPUT_FOLDER,

"X_train_under.csv"

),

index=False

)

pd.DataFrame(

y_under,

columns=["Label"]

).to_csv(

os.path.join(

OUTPUT_FOLDER,

"y_train_under.csv"

),

index=False

)

print("\nSaved")

print("X_train_under.csv")

print("y_train_under.csv")

print("\nSTEP 5A COMPLETED")