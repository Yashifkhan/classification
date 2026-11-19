
# ============================================
# 1. Import Libraries
# ============================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage


# ============================================
# 2. Load Dataset
# ============================================

file_path = "../../data/Mall_Customers.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print(df.head())


# ============================================
# 3. Select Features
# ============================================

features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

print("\nOriginal Shape:", X.shape)
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

print("\nScaled Shape:", X_scaled.shape)
print("Scaled Columns:", X_scaled.columns.tolist())


# ============================================
# 5. Create Dendrogram
# ============================================

plt.figure(figsize=(15, 7))

linked = linkage(
    X_scaled,
    method="ward"
)

dendrogram(
    linked,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=10
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Groups")
plt.ylabel("Euclidean Distance")
plt.grid(axis="y", alpha=0.3)
plt.show()


# ============================================
# 6. Train Agglomerative Clustering Model
# ============================================

hc = AgglomerativeClustering(
    n_clusters=5,
    metric="euclidean",
    linkage="ward"
)

labels = hc.fit_predict(X_scaled)

print("\nCluster Labels:")
print(labels)


# ============================================
# 7. Add Cluster Labels to Dataset
# ============================================

df["Hierarchical_Cluster"] = labels

print("\nUpdated Dataset:")
print(df.head(10))


# ============================================
# 8. Visualize Clusters
# ============================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Hierarchical_Cluster",
    palette="viridis",
    s=100
)

plt.title("Customer Segmentation using Hierarchical Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.grid(alpha=0.2)
plt.show()


# ============================================
# 9. Cluster Summary
# ============================================

cluster_summary = df.groupby("Hierarchical_Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
)

print("\nHierarchical Clustering Summary:")
print(cluster_summary.round(2))


# ============================================
# 10. Cluster Distribution
# ============================================

print("\nNumber of Customers in Each Cluster:")
print(df["Hierarchical_Cluster"].value_counts().sort_index())
