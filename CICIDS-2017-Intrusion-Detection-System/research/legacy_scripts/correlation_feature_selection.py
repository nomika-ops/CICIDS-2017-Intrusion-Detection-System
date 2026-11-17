import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
X_train_path = r"D:\IDS research\outputs\X_train.csv"
X_test_path = r"D:\IDS research\outputs\X_test.csv"

save_dir = r"D:\IDS research\feature_selection"
os.makedirs(save_dir, exist_ok=True)

print("Loading data...")

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)

print("Original Features:", X_train.shape[1])

# Correlation matrix
corr_matrix = X_train.corr().abs()

# Heatmap
plt.figure(figsize=(15,12))
sns.heatmap(corr_matrix,
            cmap="coolwarm",
            center=0)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    os.path.join(save_dir,
                 "correlation_heatmap.png")
)

plt.close()

# Upper triangle
upper = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)
)

# Features to drop
threshold = 0.90

to_drop = [
    column
    for column in upper.columns
    if any(upper[column] > threshold)
]

print("\nHighly Correlated Features:", len(to_drop))

print("\nDropped Features:")
for f in to_drop:
    print(f)

# Reduced datasets
X_train_reduced = X_train.drop(columns=to_drop)
X_test_reduced = X_test.drop(columns=to_drop)

print("\nReduced Features:",
      X_train_reduced.shape[1])

# Save
X_train_reduced.to_csv(
    os.path.join(save_dir,
                 "X_train_corr_reduced.csv"),
    index=False
)

X_test_reduced.to_csv(
    os.path.join(save_dir,
                 "X_test_corr_reduced.csv"),
    index=False
)

print("\nSaved Successfully")