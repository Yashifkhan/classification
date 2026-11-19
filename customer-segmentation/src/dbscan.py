
# ============================================
# 1. Import Libraries
# ============================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# ============================================
# 2. Load Dataset
# ============================================

file_path = "../../data/Mall_Customers.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


# ============================================
# 3. Select Features
# ============================================

features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

print("\n========== FEATURE SELECTION ==========")

print("Original Shape:", X.shape)
print("Original Columns:", X.columns.tolist())


# ============================================
# 4. Feature Scaling
# ============================================

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=features,
    index=df.index
)

print("\n========== FEATURE SCALING ==========")

print("Scaled Shape:", X_scaled.shape)
print("Scaled Columns:", X_scaled.columns.tolist())


# ============================================
# 5. K-Distance Graph
# ============================================

min_samples = 5

neighbors = NearestNeighbors(
    n_neighbors=min_samples
)

neighbors_fit = neighbors.fit(X_scaled)

distances, indices = neighbors_fit.kneighbors(X_scaled)

# Distance to the 5th nearest neighbor
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(10, 6))

plt.plot(k_distances)

plt.title("K-Distance Graph for DBSCAN")
plt.xlabel("Data Points (Sorted)")
plt.ylabel(f"Distance to {min_samples}th Nearest Neighbor")

plt.grid(alpha=0.3)
plt.show()


# ============================================
# 6. Train DBSCAN Model
# ============================================

# Initial eps value.
# Inspect the K-distance graph and tune this value.

eps_value = 0.3

dbscan = DBSCAN(
    eps=eps_value,
    min_samples=min_samples,
    metric="euclidean"
)

labels = dbscan.fit_predict(X_scaled)

print("\n========== DBSCAN MODEL ==========")

print("EPS:", eps_value)
print("Min Samples:", min_samples)
print("Cluster Labels:")
print(labels)


# ============================================
# 7. Add Cluster Labels
# ============================================

df["DBSCAN_Cluster"] = labels

print("\n========== UPDATED DATASET ==========")

print(df.head(10))


# ============================================
# 8. Visualize DBSCAN Clusters
# ============================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="DBSCAN_Cluster",
    palette="viridis",
    s=100
)

plt.title("Customer Segmentation using DBSCAN")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.grid(alpha=0.2)
plt.show()


# ============================================
# 9. Cluster Count and Noise Detection
# ============================================

unique_labels = set(labels)

n_clusters = len(unique_labels - {-1})

n_noise = np.sum(labels == -1)

print("\n========== CLUSTER INFORMATION ==========")

print("Number of Clusters:", n_clusters)
print("Number of Noise Points:", n_noise)

print("\nCluster Distribution:")

print(
    df["DBSCAN_Cluster"]
    .value_counts()
    .sort_index()
)


# ============================================
# 10. Identify Core, Border and Noise Points
# ============================================

core_indices = dbscan.core_sample_indices_

core_mask = np.zeros(len(X_scaled), dtype=bool)

core_mask[core_indices] = True

df["Point_Type"] = np.where(
    core_mask,
    "Core",
    np.where(
        labels == -1,
        "Noise",
        "Border"
    )
)

print("\n========== POINT TYPE DISTRIBUTION ==========")

print(df["Point_Type"].value_counts())


# ============================================
# 11. Visualize Core, Border and Noise Points
# ============================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Point_Type",
    style="Point_Type",
    palette={
        "Core": "green",
        "Border": "orange",
        "Noise": "red"
    },
    s=100
)

plt.title("DBSCAN - Core, Border and Noise Points")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Point Type")
plt.grid(alpha=0.2)
plt.show()


# ============================================
# 12. Cluster Summary
# ============================================

cluster_summary = df.groupby("DBSCAN_Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
)

print("\n========== DBSCAN CLUSTER SUMMARY ==========")

print(cluster_summary.round(2))


# ============================================
# 13. Final Results
# ============================================

print("\n========== FINAL RESULTS ==========")

print("Total Customers:", len(df))
print("Total Clusters:", n_clusters)
print("Total Noise Points:", n_noise)

print("Core Points:", np.sum(df["Point_Type"] == "Core"))
print("Border Points:", np.sum(df["Point_Type"] == "Border"))
print("Noise Points:", np.sum(df["Point_Type"] == "Noise"))
