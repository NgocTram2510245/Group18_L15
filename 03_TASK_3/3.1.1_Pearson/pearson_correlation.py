import pandas as pd
import numpy as np
from pathlib import Path


# 1. DEFINE DATA PATHS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "00_DU_LIEU_CHUNG" / "data" / "processed"

TRAIN_PATH = DATA_DIR / "train_clean.csv"
DICTIONARY_PATH = DATA_DIR / "data_dictionary.csv"


# 2. LOAD DATA


df = pd.read_csv(TRAIN_PATH)
data_dictionary = pd.read_csv(DICTIONARY_PATH)

print("Dataset shape:", df.shape)
print("Data dictionary shape:", data_dictionary.shape)


# 3. CHECK DATA


print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes.value_counts())


# 4. SELECT NUMERICAL FEATURES


categorical_cols = [
    "protocol_type",
    "service",
    "flag"
]

metadata_cols = [
    "attack_name",
    "difficulty",
    "attack_category",
    "label_binary"
]

exclude_cols = categorical_cols + metadata_cols

numeric_df = df.select_dtypes(
    include=["number"]
).drop(
    columns=exclude_cols,
    errors="ignore"
)


# 5. REMOVE CONSTANT FEATURES


constant_cols = [
    col for col in numeric_df.columns
    if numeric_df[col].nunique() <= 1
]

numeric_df = numeric_df.drop(columns=constant_cols)

# 6. DISPLAY RESULTS


print("\nNumber of numerical features:", numeric_df.shape[1])

print("\nConstant features removed:")
print(constant_cols)

print("\nFeatures selected for Pearson:")
print(numeric_df.columns.tolist())

print("\nPrepared dataset shape:", numeric_df.shape)

# 7. CALCULATE PEARSON CORRELATION MATRIX


pearson_matrix = numeric_df.corr(method="pearson")

print("\nPearson Correlation Matrix:")
print(pearson_matrix)


# 8. SAVE PEARSON MATRIX


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pearson_matrix.to_csv(
    OUTPUT_DIR / "pearson_matrix.csv"
)

print("\nPearson matrix saved successfully!")


# 9. VALIDATE PEARSON MATRIX


print("\nMatrix shape:", pearson_matrix.shape)

print(
    "Number of correlation pairs:",
    len(pearson_matrix.columns) * (len(pearson_matrix.columns) - 1) // 2
)

print(
    "Contains NaN:",
    pearson_matrix.isna().any().any()
)

print(
    "Contains infinite values:",
    np.isinf(pearson_matrix.to_numpy()).any()
)




# 10. VISUALIZE PEARSON MATRIX USING HEATMAP


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18, 15))

sns.heatmap(
    pearson_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    annot=False,
    square=True,
    linewidths=0.3,
    cbar_kws={
        "label": "Pearson Correlation Coefficient"
    }
)

plt.title(
    "Pearson Correlation Heatmap",
    fontsize=18,
    pad=20
)

plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pearson_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nHeatmap saved successfully!")





# 11. EXTRACT HIGHLY CORRELATED PAIRS


THRESHOLD = 0.95

high_correlation_pairs = []

# Chỉ duyệt tam giác trên của ma trận
# để tránh lấy trùng cặp A-B và B-A

for i in range(len(pearson_matrix.columns)):
    for j in range(i + 1, len(pearson_matrix.columns)):

        feature_1 = pearson_matrix.columns[i]
        feature_2 = pearson_matrix.columns[j]

        correlation = pearson_matrix.iloc[i, j]

        if abs(correlation) > THRESHOLD:
            high_correlation_pairs.append({
                "Feature 1": feature_1,
                "Feature 2": feature_2,
                "Pearson Correlation": correlation,
                "Absolute Correlation": abs(correlation)
            })

# Chuyển danh sách thành DataFrame
high_corr_df = pd.DataFrame(high_correlation_pairs)

# Sắp xếp theo độ mạnh tương quan giảm dần
high_corr_df = high_corr_df.sort_values(
    by="Absolute Correlation",
    ascending=False
)

# Lưu kết quả
high_corr_df.to_csv(
    OUTPUT_DIR / "high_correlation_pairs.csv",
    index=False
)

print("\nHighly Correlated Feature Pairs:")
print(high_corr_df.to_string(index=False))

print("\nNumber of highly correlated pairs:", len(high_corr_df))

print("\nHigh correlation pairs saved successfully!")