import pandas as pd
import os
import numpy as np
from pathlib import Path
folder = r"D:\IDS research\MachineLearningCVE"

files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

dfs = []
for f in files:
    path = os.path.join(folder, f)
    if os.path.exists(path):
        print("Loading:", path)
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
    else:
        print("Missing:", path)

data = pd.concat(dfs, ignore_index=True)
data.columns = data.columns.str.strip()
print("="*70)
print("DATASET OVERVIEW")
print("="*70)

print(f"Shape: {data.shape}")

print(f"\nNumber of Features: {data.shape[1]}")

print(f"Number of Records: {data.shape[0]}")

print("\nColumn Names:")
print(data.columns.tolist())

print("\nData Types:")
print(data.dtypes)

print("\nMemory Usage (MB):")
print(round(data.memory_usage(deep=True).sum()/1024**2,2))
print("\n" + "="*70)
print("DATA QUALITY ASSESSMENT")
print("="*70)

# -----------------------------
# Missing Values
# -----------------------------
missing = data.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

print("\nColumns with Missing Values:")

if len(missing) == 0:
    print("No missing values found.")
else:
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": (missing / len(data)) * 100
    })

    print(missing_df)

# ------------------------------------------------------------------
# INFINITE VALUE ANALYSIS
# ------------------------------------------------------------------

print("\n" + "="*70)
print("INFINITE VALUE ANALYSIS")
print("="*70)

# Get numeric columns without using select_dtypes (avoids memory issue)
numeric_columns = [
    col for col in data.columns
    if pd.api.types.is_numeric_dtype(data[col])
]

positive_inf = []
negative_inf = []

for col in numeric_columns:
    values = data[col].to_numpy()

    positive_inf.append(np.isposinf(values).sum())
    negative_inf.append(np.isneginf(values).sum())

inf_per_column = pd.DataFrame({
    "Positive Infinity": positive_inf,
    "Negative Infinity": negative_inf
}, index=numeric_columns)

inf_per_column["Total"] = (
    inf_per_column["Positive Infinity"] +
    inf_per_column["Negative Infinity"]
)

inf_per_column = inf_per_column[inf_per_column["Total"] > 0]

if len(inf_per_column) == 0:
    print("No infinite values found.")
else:
    print(inf_per_column.sort_values(by="Total", ascending=False))

print("\n" + "="*70)
print("TARGET VARIABLE ANALYSIS")
print("="*70)

label_counts = data["Label"].value_counts(dropna=False)

print(f"\nNumber of Classes: {len(label_counts)}")

print("\nClass Distribution:\n")

target_report = []

for label, count in label_counts.items():

    # Convert problematic unicode characters into printable text
    safe_label = str(label).encode("ascii", errors="replace").decode("ascii")

    percentage = (count / len(data)) * 100

    print(f"{safe_label:<35} {count:>10} ({percentage:.4f}%)")

    target_report.append({
        "Label": safe_label,
        "Count": count,
        "Percentage": round(percentage, 4)
    })

target_report = pd.DataFrame(target_report)

# Save report
target_report.to_csv("target_distribution.csv", index=False)

print("\nTarget distribution saved as target_distribution.csv")
# ============================================================
# PHASE 2 : DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2 : DATA CLEANING")
print("=" * 70)

# ------------------------------------------------------------
# Create Output Folder
# ------------------------------------------------------------

OUTPUT_DIR = Path(r"D:\IDS research\cicids_project\EDA_Outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Store Original Dataset Information
# ------------------------------------------------------------

original_rows = len(data)
original_columns = data.shape[1]

print(f"\nOriginal Shape : {data.shape}")

# ------------------------------------------------------------
# STEP 1 : Count Infinite Values
# ------------------------------------------------------------

numeric_columns = data.select_dtypes(include=np.number)

total_infinite = np.isinf(numeric_columns).sum().sum()

print(f"\nInfinite Values Found : {total_infinite}")

# ------------------------------------------------------------
# STEP 2 : Replace Infinite Values
# ------------------------------------------------------------

data = data.replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# STEP 3 : Missing Values
# ------------------------------------------------------------

missing_after_inf = data.isnull().sum().sum()

print(f"Missing Values : {missing_after_inf}")

rows_before_missing = len(data)

data = data.dropna()

rows_after_missing = len(data)

rows_removed_missing = rows_before_missing - rows_after_missing

print(f"Rows Removed (Missing Values) : {rows_removed_missing}")

# ------------------------------------------------------------
# STEP 4 : Duplicate Rows
# ------------------------------------------------------------

duplicate_rows = data.duplicated().sum()

print(f"\nDuplicate Rows Found : {duplicate_rows}")

rows_before_duplicates = len(data)

data = data.drop_duplicates()

rows_after_duplicates = len(data)

rows_removed_duplicates = rows_before_duplicates - rows_after_duplicates

print(f"Duplicate Rows Removed : {rows_removed_duplicates}")

# ------------------------------------------------------------
# STEP 5 : Standardize Label Names
# ------------------------------------------------------------

label_mapping = {
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack � Sql Injection": "Web Attack - Sql Injection",
    "Web Attack ? Brute Force": "Web Attack - Brute Force",
    "Web Attack ? XSS": "Web Attack - XSS",
    "Web Attack ? Sql Injection": "Web Attack - Sql Injection"
}

data["Label"] = data["Label"].replace(label_mapping)

# ------------------------------------------------------------
# STEP 6 : Verify Labels
# ------------------------------------------------------------

print("\nLabels After Cleaning\n")

for label in sorted(data["Label"].unique()):
    print(label)

# ------------------------------------------------------------
# STEP 7 : Final Dataset Information
# ------------------------------------------------------------

print("\n" + "-" * 60)

print(f"Final Shape : {data.shape}")

rows_removed_total = original_rows - len(data)

print(f"Total Rows Removed : {rows_removed_total}")

print(f"Percentage Removed : {(rows_removed_total/original_rows)*100:.4f}%")

print("-" * 60)

# ------------------------------------------------------------
# STEP 8 : Save Cleaned Dataset
# ------------------------------------------------------------

cleaned_dataset = OUTPUT_DIR / "cleaned_cicids2017.csv"

data.to_csv(cleaned_dataset, index=False)

print(f"\nCleaned Dataset Saved")

# ------------------------------------------------------------
# STEP 9 : Save Preprocessing Report
# ------------------------------------------------------------

summary = pd.DataFrame({

    "Metric":[
        "Original Rows",
        "Original Columns",
        "Infinite Values",
        "Missing Values",
        "Rows Removed (Missing)",
        "Duplicate Rows Removed",
        "Final Rows",
        "Final Columns"
    ],

    "Value":[
        original_rows,
        original_columns,
        total_infinite,
        missing_after_inf,
        rows_removed_missing,
        rows_removed_duplicates,
        len(data),
        data.shape[1]
    ]

})

summary.to_csv(OUTPUT_DIR / "preprocessing_summary.csv", index=False)

print("Preprocessing Summary Saved")

print("\nPHASE 2 COMPLETED SUCCESSFULLY")
# ============================================================
# PHASE 3 : FEATURE SCREENING
# ============================================================

print("\n" + "="*70)
print("PHASE 3 : FEATURE SCREENING")
print("="*70)

# ------------------------------------------------------------
# Feature Types
# ------------------------------------------------------------

numerical_features = data.select_dtypes(include=np.number).columns.tolist()
categorical_features = data.select_dtypes(exclude=np.number).columns.tolist()

print(f"\nTotal Features      : {data.shape[1]}")
print(f"Numerical Features  : {len(numerical_features)}")
print(f"Categorical Features: {len(categorical_features)}")

# ------------------------------------------------------------
# Unique Values
# ------------------------------------------------------------

print("\nCalculating unique values...")

unique_counts = data.nunique()

feature_summary = pd.DataFrame({
    "Feature": unique_counts.index,
    "Unique Values": unique_counts.values,
    "Data Type": data.dtypes.astype(str).values
})

feature_summary = feature_summary.sort_values(
    by="Unique Values"
)

feature_summary.to_csv(
    OUTPUT_DIR / "feature_summary.csv",
    index=False
)

# ------------------------------------------------------------
# Constant Features
# ------------------------------------------------------------

constant_features = feature_summary[
    feature_summary["Unique Values"] == 1
]

print(f"\nConstant Features : {len(constant_features)}")

if len(constant_features):

    print(constant_features["Feature"].tolist())

constant_features.to_csv(
    OUTPUT_DIR / "constant_features.csv",
    index=False
)

# ------------------------------------------------------------
# Near Constant Features
# ------------------------------------------------------------

print("\nSearching for near-constant features...")

near_constant = []

threshold = 0.99

for column in numerical_features:

    freq = data[column].value_counts(normalize=True)

    if len(freq):

        if freq.iloc[0] >= threshold:

            near_constant.append({

                "Feature": column,

                "Most Common Value %": round(freq.iloc[0]*100,4)

            })

near_constant = pd.DataFrame(near_constant)

print(f"Near Constant Features : {len(near_constant)}")

near_constant.to_csv(
    OUTPUT_DIR / "near_constant_features.csv",
    index=False
)

# ------------------------------------------------------------
# Duplicate Columns
# ------------------------------------------------------------

print("\nChecking duplicate columns...")

duplicate_columns = []

columns = data.columns

for i in range(len(columns)):

    for j in range(i+1, len(columns)):

        col1 = columns[i]
        col2 = columns[j]

        if data[col1].equals(data[col2]):

            duplicate_columns.append([col1, col2])

duplicate_df = pd.DataFrame(
    duplicate_columns,
    columns=["Column 1","Column 2"]
)

print(f"Duplicate Columns : {len(duplicate_df)}")

duplicate_df.to_csv(
    OUTPUT_DIR / "duplicate_columns.csv",
    index=False
)

# ------------------------------------------------------------
# Zero Variance Features
# ------------------------------------------------------------

variance = data[numerical_features].var()

zero_variance = variance[
    variance == 0
]

print(f"\nZero Variance Features : {len(zero_variance)}")

zero_variance.to_csv(
    OUTPUT_DIR / "zero_variance_features.csv"
)

print("\nFeature Screening Report Saved")

print("\nPHASE 3 COMPLETED SUCCESSFULLY")
# ============================================================
# PHASE 4 : STATISTICAL FEATURE PROFILING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4 : STATISTICAL FEATURE PROFILING")
print("=" * 70)

numerical_features = data.select_dtypes(include=np.number).columns

stats = pd.DataFrame(index=numerical_features)

stats["Mean"] = data[numerical_features].mean()
stats["Median"] = data[numerical_features].median()
stats["Std"] = data[numerical_features].std()
stats["Variance"] = data[numerical_features].var()
stats["Min"] = data[numerical_features].min()
stats["Max"] = data[numerical_features].max()

stats["Skewness"] = data[numerical_features].skew()
stats["Kurtosis"] = data[numerical_features].kurt()

stats["Unique Values"] = data[numerical_features].nunique()

stats = stats.round(4)

stats.to_csv(OUTPUT_DIR / "statistical_summary.csv")

print("\nStatistical summary saved.")

# ------------------------------------------------------------
# Top Skewed Features
# ------------------------------------------------------------

top_skew = stats.reindex(
    stats["Skewness"].abs().sort_values(ascending=False).index
)

top_skew.head(15).to_csv(
    OUTPUT_DIR / "top_skewed_features.csv"
)

print("\nTop 15 most skewed features:")

print(top_skew[["Skewness"]].head(15))

# ------------------------------------------------------------
# Heavy Tailed Features
# ------------------------------------------------------------

heavy_tail = stats[
    stats["Kurtosis"] > 10
].sort_values(
    by="Kurtosis",
    ascending=False
)

heavy_tail.to_csv(
    OUTPUT_DIR / "heavy_tailed_features.csv"
)

print(f"\nHeavy-tailed features (>10 kurtosis): {len(heavy_tail)}")

print("\nTop Heavy-tailed Features")

print(
    heavy_tail[["Kurtosis"]].head(15)
)

print("\nPHASE 4 COMPLETED SUCCESSFULLY")
# ============================================================
# PHASE 5 : CORRELATION ANALYSIS
# ============================================================

print("\n" + "="*70)
print("PHASE 5 : CORRELATION ANALYSIS")
print("="*70)

# ------------------------------------------------------------
# STEP 1 : Remove Constant Features
# ------------------------------------------------------------

constant_features = [
    "Bwd PSH Flags",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bytes/Bulk",
    "Bwd Avg Bulk Rate",
    "Bwd URG Flags"
]

corr_data = data.drop(columns=constant_features)

print(f"\nRemoved Constant Features : {len(constant_features)}")
print(f"Remaining Features : {corr_data.shape[1]}")

# ------------------------------------------------------------
# STEP 2 : Correlation Matrix
# ------------------------------------------------------------

print("\nCalculating Pearson Correlation Matrix...")

corr_matrix = corr_data.select_dtypes(include=np.number).corr()

corr_matrix.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

print("Correlation Matrix Saved.")

# ------------------------------------------------------------
# STEP 3 : Extract Highly Correlated Pairs
# ------------------------------------------------------------

thresholds = [0.90, 0.95, 0.99]

for threshold in thresholds:

    pairs = []

    columns = corr_matrix.columns

    for i in range(len(columns)):

        for j in range(i+1, len(columns)):

            corr_value = corr_matrix.iloc[i, j]

            if abs(corr_value) >= threshold:

                pairs.append([
                    columns[i],
                    columns[j],
                    round(corr_value,4)
                ])

    pair_df = pd.DataFrame(
        pairs,
        columns=[
            "Feature 1",
            "Feature 2",
            "Correlation"
        ]
    )

    pair_df = pair_df.sort_values(
        by="Correlation",
        ascending=False,
        key=abs
    )

    filename = f"correlated_pairs_{str(threshold).replace('.','')}.csv"

    pair_df.to_csv(
        OUTPUT_DIR / filename,
        index=False
    )

    print(f"\nThreshold {threshold}")
    print(f"Highly Correlated Pairs : {len(pair_df)}")

# ------------------------------------------------------------
# STEP 4 : Heatmap (Clean Features Only)
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20,16))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    square=False,
    cbar=True
)

plt.title("Correlation Heatmap (After Removing Constant Features)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.close()

print("\nHeatmap Saved.")

print("\nPHASE 5 COMPLETED SUCCESSFULLY")
# ============================================================
# PHASE 6 : DATA VISUALIZATION
# ============================================================

print("\n" + "="*70)
print("PHASE 6 : DATA VISUALIZATION")
print("="*70)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------------------------------------------
# Create sample ONLY for visualization
# ------------------------------------------------------------

print("\nCreating visualization sample (100000 rows)...")

viz_data = data.sample(n=100000, random_state=42)

top_features = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Packet Length Mean",
    "Average Packet Size",
    "Destination Port"
]

# ============================================================
# HISTOGRAM GRID (LOG SCALE)
# ============================================================

print("Generating Histogram Grid...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, feature in enumerate(top_features):

    plot_data = viz_data[feature].replace([np.inf, -np.inf], np.nan)
    plot_data = plot_data.dropna()
    plot_data = plot_data[plot_data >= 0]
    plot_data = np.log1p(plot_data)

    sns.histplot(
        plot_data,
        bins=50,
        kde=True,
        ax=axes[i]
    )

    axes[i].set_title(feature, fontsize=10)
    axes[i].set_xlabel("log(1+x)", fontsize=8)
    axes[i].set_ylabel("")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "histogram_grid_log.png",
    dpi=300
)

plt.close()

# ============================================================
# BOXPLOT GRID (LOG SCALE)
# ============================================================

print("Generating Boxplot Grid...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, feature in enumerate(top_features):

    plot_data = viz_data[feature].replace([np.inf, -np.inf], np.nan)
    plot_data = plot_data.dropna()
    plot_data = plot_data[plot_data >= 0]
    plot_data = np.log1p(plot_data)

    sns.boxplot(
        x=plot_data,
        ax=axes[i]
    )

    axes[i].set_title(feature, fontsize=10)
    axes[i].set_xlabel("log(1+x)", fontsize=8)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "boxplot_grid_log.png",
    dpi=300
)

plt.close()

print("\n" + "-"*60)
print("Visualization Files Generated")
print("- histogram_grid_log.png")
print("- boxplot_grid_log.png")
print("- Existing class_distribution.png")
print("- Existing correlation_heatmap.png")
print("- Existing pca_visualization.png")
print("-"*60)

print("\nPHASE 6 COMPLETED SUCCESSFULLY")
print(type(data))
# ======================================================================
# PHASE 7 : FEATURE VS TARGET ANALYSIS
# ======================================================================

print("\n" + "="*70)
print("PHASE 7 : FEATURE VS TARGET ANALYSIS")
print("="*70)

import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Use cleaned dataframe
# ------------------------------------------------------------

data = cleaned_data if "cleaned_data" in globals() else data

# ------------------------------------------------------------
# Binary Target
# ------------------------------------------------------------

data["Target"] = np.where(data["Label"]=="BENIGN","BENIGN","ATTACK")

# ------------------------------------------------------------
# Output Folder
# ------------------------------------------------------------

phase7_folder = OUTPUT_DIR / "Phase7_Target_Analysis"
phase7_folder.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Important Features
# ------------------------------------------------------------

features = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Packet Length Mean",
    "Average Packet Size",
    "Destination Port"
]

# ============================================================
# Statistical Summary
# ============================================================

print("\nCreating Statistical Summary...")

rows=[]

for feature in features:

    benign=data.loc[data["Target"]=="BENIGN",feature]
    attack=data.loc[data["Target"]=="ATTACK",feature]

    rows.append({

        "Feature":feature,

        "BENIGN Mean":benign.mean(),
        "ATTACK Mean":attack.mean(),

        "BENIGN Median":benign.median(),
        "ATTACK Median":attack.median(),

        "BENIGN Std":benign.std(),
        "ATTACK Std":attack.std(),

        "BENIGN Min":benign.min(),
        "ATTACK Min":attack.min(),

        "BENIGN Max":benign.max(),
        "ATTACK Max":attack.max()

    })

stats=pd.DataFrame(rows)

stats.to_csv(
    phase7_folder/"feature_target_statistics.csv",
    index=False
)

print("Statistical Summary Saved.")

# ============================================================
# KDE GRID
# ============================================================

print("Creating KDE Grid...")

fig,axes=plt.subplots(2,4,figsize=(18,10))

axes=axes.flatten()

sample=data.sample(100000,random_state=42)

for ax,feature in zip(axes,features):

    sns.kdeplot(
        data=sample,
        x=feature,
        hue="Target",
        fill=True,
        common_norm=False,
        ax=ax
    )

    ax.set_title(feature)

plt.tight_layout()

plt.savefig(
    phase7_folder/"feature_vs_target_kde_grid.png",
    dpi=300
)

plt.close()

# ============================================================
# Violin Plot Grid
# ============================================================

print("Creating Violin Plots...")

fig,axes=plt.subplots(2,4,figsize=(18,10))

axes=axes.flatten()

for ax,feature in zip(axes,features):

    sns.violinplot(
        data=sample,
        x="Target",
        y=feature,
        ax=ax,
        cut=0
    )

    ax.set_title(feature)

plt.tight_layout()

plt.savefig(
    phase7_folder/"feature_vs_target_violin.png",
    dpi=300
)

plt.close()

# ============================================================
# Pair Plot
# ============================================================

print("Creating Pair Plot...")

pair_sample=sample[[
    "Flow Duration",
    "Flow Bytes/s",
    "Packet Length Mean",
    "Average Packet Size",
    "Target"
]].sample(3000,random_state=42)

pair=sns.pairplot(
    pair_sample,
    hue="Target",
    corner=True,
    diag_kind="kde"
)

pair.savefig(
    phase7_folder/"feature_target_pairplot.png",
    dpi=300
)

plt.close()

# ============================================================
# Mean Comparison Bar Chart
# ============================================================

print("Creating Mean Comparison...")

means=[]

for feature in features:

    means.append({
        "Feature":feature,
        "BENIGN":data.loc[data.Target=="BENIGN",feature].mean(),
        "ATTACK":data.loc[data.Target=="ATTACK",feature].mean()
    })

means=pd.DataFrame(means)

means.set_index("Feature").plot(
    kind="bar",
    figsize=(12,6)
)

plt.ylabel("Mean")

plt.tight_layout()

plt.savefig(
    phase7_folder/"feature_mean_comparison.png",
    dpi=300
)

plt.close()

print("\n------------------------------------------------")
print("Generated Files")
print("------------------------------------------------")
print("feature_target_statistics.csv")
print("feature_vs_target_kde_grid.png")
print("feature_vs_target_violin.png")
print("feature_target_pairplot.png")
print("feature_mean_comparison.png")
print("------------------------------------------------")

print("\nPHASE 7 COMPLETED SUCCESSFULLY")