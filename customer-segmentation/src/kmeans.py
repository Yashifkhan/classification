import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
file_path="../../data/Mall_Customers.csv"
df=pd.read_csv(file_path)
# print(df.head())

features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(df[features]),
    columns=features,
    index=df.index
)

print("Original Shape:", df[features].shape)
print("Scaled Shape:", X_scaled.shape)

print("Original Columns:", df[features].columns.tolist())
print("Scaled Columns:", X_scaled.columns.tolist())


kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    n_init=10,
    random_state=42
)

kmeans.fit(X_scaled)
labels = kmeans.labels_

print(labels)
df["Cluster"] = labels

print(df.head(10))
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Customer Segmentation using K-Means")
plt.legend(title="Cluster")

plt.show()