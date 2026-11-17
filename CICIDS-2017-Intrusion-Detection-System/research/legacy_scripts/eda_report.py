import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# ==========================
# LOAD DATA
# ==========================

DATA_PATH = r"D:\IDS research\outputs\X_train.csv"
LABEL_PATH = r"D:\IDS research\outputs\y_train.csv"

X = pd.read_csv(DATA_PATH)
y = pd.read_csv(LABEL_PATH)

df = pd.concat([X, y], axis=1)

output_dir = "eda_outputs"
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("DATASET OVERVIEW")
print("="*60)

print("Shape:", df.shape)
print("\nTarget Distribution:")
print(df["Label"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nData Types:")
print(df.dtypes.value_counts())

# ==========================
# 1. CLASS DISTRIBUTION
# ==========================

plt.figure(figsize=(6,4))
df["Label"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/class_distribution.png")
plt.close()

print("Saved: class_distribution.png")

# ==========================
# 2. MISSING VALUES
# ==========================

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) > 0:
    plt.figure(figsize=(12,5))
    missing.sort_values(ascending=False).plot(kind="bar")
    plt.title("Missing Values")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/missing_values.png")
    plt.close()

# ==========================
# 3. SUMMARY STATISTICS
# ==========================

summary = df.describe()

summary.to_csv(
    f"{output_dir}/summary_statistics.csv"
)

print("Saved: summary_statistics.csv")

# ==========================
# 4. CORRELATION HEATMAP
# ==========================

sample_cols = X.columns[:20]

corr = df[sample_cols].corr()

plt.figure(figsize=(12,8))

sns.heatmap(
    corr,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap (Top 20 Features)")
plt.tight_layout()

plt.savefig(
    f"{output_dir}/correlation_heatmap.png"
)

plt.close()

print("Saved: correlation_heatmap.png")

# ==========================
# 5. FEATURE DISTRIBUTIONS
# ==========================

top_features = X.columns[:4]

for col in top_features:

    plt.figure(figsize=(7,4))

    df[col].hist(
        bins=50
    )

    plt.title(col)

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/{col}_distribution.png"
    )

    plt.close()

print("Saved Feature Distributions")

# ==========================
# 6. PCA VISUALIZATION
# ==========================

sample_size = min(5000, len(df))

sample = df.sample(
    sample_size,
    random_state=42
)

X_sample = sample.drop(
    "Label",
    axis=1
)

y_sample = sample["Label"]

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_sample
)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y_sample,
    alpha=0.6
)

plt.title("PCA Visualization")

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/pca_visualization.png"
)

plt.close()

print("Saved: pca_visualization.png")

# ==========================
# 7. BOXPLOTS
# ==========================

for col in top_features:

    plt.figure(figsize=(7,4))

    sns.boxplot(
        x="Label",
        y=col,
        data=df.sample(
            min(10000, len(df)),
            random_state=42
        )
    )

    plt.title(
        f"{col} vs Label"
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/{col}_boxplot.png"
    )

    plt.close()

print("Saved Boxplots")

# ==========================
# COMPLETE
# ==========================

print("\n" + "="*60)
print("EDA COMPLETE")
print("="*60)

print(f"\nOutputs saved in: {output_dir}")

print("""
Generated Files:
1. class_distribution.png
2. summary_statistics.csv
3. correlation_heatmap.png
4. feature distributions
5. PCA visualization
6. Boxplots
""")