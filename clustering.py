# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =========================
# STEP 1: LOAD CSV FILE
# =========================

print("Loading survey data...")

df = pd.read_csv("studentdata_ordinal_kmeans.csv", encoding="latin1")

df.columns = df.columns.str.strip()

print("\nColumns found:")
print(df.columns)

# =========================
# STEP 2: SELECT NUMERIC DATA
# =========================

data = df.select_dtypes(include=[np.number])

print("\nNumeric data preview BEFORE cleaning:")
print(data.head())

# =========================
# STEP 3: HANDLE MISSING VALUES
# =========================

data = data.fillna(data.mean())

print("\nMissing values after cleaning:")
print(data.isna().sum())

# =========================
# STEP 4: ELBOW METHOD
# =========================

print("\nRunning Elbow Method...")

inertia_values = []
k_range = range(1, 7)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data)
    inertia_values.append(kmeans.inertia_)

plt.figure()
plt.plot(k_range, inertia_values, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("elbow_graph.png")
plt.show()

print("Elbow graph saved as elbow_graph.png")

# =========================
# STEP 5: FINAL CLUSTERING
# =========================

print("\nRunning Final K-Means Clustering...")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(data)

clusters = kmeans.labels_

# Add cluster column
df["Cluster"] = clusters

# Save clustered file
df.to_csv("studentdata_clustered.csv", index=False)

print("\nClustered file saved as studentdata_clustered.csv")

# =========================
# STEP 6: COUNT DATA PER CLUSTER
# =========================

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\nNumber of respondents per cluster:")

for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} respondents")

# =========================
# STEP 7: PIE CHART (CIRCLE GRAPH)
# =========================

plt.figure()

plt.pie(
    cluster_counts,
    labels=[f"Cluster {i}" for i in cluster_counts.index],
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Distribution of Respondents per Cluster")

plt.savefig("cluster_pie_chart.png")

plt.show()

print("Pie chart saved as cluster_pie_chart.png")

# =========================
# STEP 8: SHOW CLUSTER CENTERS
# =========================

print("\nCluster Centers:")

centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=data.columns
)

print(centers)

print("\nSUCCESS: Clustering and visualization complete.")