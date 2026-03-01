import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler




print("Loading dataset...")

df = pd.read_csv("Old/studentdata_ordinal_kmeans.csv", encoding="latin1")
df.columns = df.columns.str.strip()
data = df.select_dtypes(include=[np.number])
data = data.fillna(data.mean())
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print("Data successfully prepared and scaled.\n")




# ELBOW METHOD 
k_values = range(1, 11)
inertia_values = []

print("Running Elbow Method...\n")

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(data_scaled)
    inertia_values.append(kmeans.inertia_)
    print(f"k = {k}, Inertia = {kmeans.inertia_:.2f}")


inertia_scaled = MinMaxScaler().fit_transform(
    np.array(inertia_values).reshape(-1, 1)
).flatten()

plt.figure(figsize=(7, 5))
plt.plot(k_values, inertia_scaled, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Normalized Inertia (SSE)")
plt.title("Elbow Method (Normalized Scale)")
plt.xticks(k_values)
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method_normalized.png", dpi=300)
plt.show()

print("Saved: elbow_method_normalized.png\n")






# SILHOUETTE METHOD
print("Running Silhouette Method...\n")

silhouette_values = []
k_silhouette = range(2, 11)

for k in k_silhouette:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    silhouette_values.append(score)
    print(f"k = {k}, Silhouette Score = {score:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(k_silhouette, silhouette_values, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method (k = 2–10)")
plt.xticks(k_silhouette)
plt.grid(True)
plt.tight_layout()
plt.savefig("silhouette_method.png", dpi=300)
plt.show()

print("Saved: silhouette_method.png\n")





# FINAL K-MEANS CLUSTERING
optimal_k = 3
print(f"Applying Final K-Means Clustering (k = {optimal_k})...\n")

kmeans_final = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans_final.fit_predict(data_scaled)

df["Cluster"] = clusters
df.to_csv("studentdata_clustered.csv", index=False)

print("Saved: studentdata_clustered.csv\n")


centers = pd.DataFrame(
    kmeans_final.cluster_centers_,
    columns=data.columns
)

print("Cluster Centers (Scaled):")
print(centers)






# PIE CHART 
cluster_counts = df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(6, 6))
plt.pie(
    cluster_counts,
    labels=[f"Cluster {i}" for i in cluster_counts.index],
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Cluster Distribution")
plt.tight_layout()
plt.savefig("cluster_distribution.png", dpi=300)
plt.show()

print("Saved: cluster_distribution.png\n")




# ITERATION VISUALIZATION 

print("Generating Iteration Visualization...\n")

X = data_scaled[:, :2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
iterations_to_show = [2, 4, 6]

for ax, iteration in zip(axes, iterations_to_show):

    kmeans_iter = KMeans(
        n_clusters=3,
        init='random',
        n_init=1,
        max_iter=iteration,
        random_state=42
    )

    kmeans_iter.fit(X)

    labels = kmeans_iter.labels_
    centers = kmeans_iter.cluster_centers_
    sse = kmeans_iter.inertia_

    ax.scatter(X[:, 0], X[:, 1], c=labels)
    ax.scatter(centers[:, 0], centers[:, 1], marker='+', s=200)
    ax.set_title(f"Iteration {iteration}, SSE={sse:.2f}")

plt.tight_layout()
plt.savefig("kmeans_iterations.png", dpi=300)
plt.show()

print("Saved: kmeans_iterations.png")
print("\nSUCCESS: All processes completed.")