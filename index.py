import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



df = pd.read_csv("datacluster_ordinal_encoded.csv")
X = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



#ELBOW METHOD
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()



# 4. SILHOUETTE METHOD
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure()
plt.plot(K_range, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()

best_k = K_range[np.argmax(silhouette_scores)]
print("\nOptimal k based on Silhouette Score:", best_k)
print("Best Silhouette Score:", max(silhouette_scores))



#kmeans
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_final.fit(X_scaled)
inertia = kmeans_final.inertia_
clusters = kmeans_final.labels_
df["Cluster"] = clusters

print("\n======================================")
print("FINAL K-MEANS RESULTS")
print("======================================")
print("Optimal k:", best_k)
print("Inertia:", inertia)
print("Number of Iterations:", kmeans_final.n_iter_)



centroids = pd.DataFrame(
    kmeans_final.cluster_centers_,
    columns=X.columns
)

print("\nCluster Centroids (Standardized Values):")
print(centroids)



# DATA DISTRIBUTION PER CLUSTER
print("\n======================================")
print("FULL DATA DISTRIBUTION PER CLUSTER")
print("======================================")

for cluster_id in sorted(df["Cluster"].unique()):
    
    cluster_data = df[df["Cluster"] == cluster_id]
    
    print("\n--------------------------------------")
    print(f"CLUSTER {cluster_id}")
    print("--------------------------------------")
    print(f"Total Data Points: {len(cluster_data)}\n")
    
    print(cluster_data.to_string(index=False))  

print("\n======================================")
print("ALL DATA HAS BEEN DISPLAYED.")
print("======================================")


#for iteration
k = 3
max_iterations = 100
tolerance = 1e-4

np.random.seed(42)
random_indices = np.random.choice(len(X_scaled), k, replace=False)
centroids_manual = X_scaled[random_indices]

print("\nInitial Centroids:\n", centroids_manual)

for iteration in range(max_iterations):

    distances = np.linalg.norm(X_scaled[:, np.newaxis] - centroids_manual, axis=2)
    labels_manual = np.argmin(distances, axis=1)

    new_centroids = np.array([
        X_scaled[labels_manual == i].mean(axis=0)
        for i in range(k)
    ])

    sse = 0
    for i in range(k):
        cluster_points = X_scaled[labels_manual == i]
        sse += np.sum((cluster_points - new_centroids[i])**2)

    print(f"Iteration {iteration+1} | SSE = {round(sse, 4)}")

    if np.all(np.abs(new_centroids - centroids_manual) < tolerance):
        print(f"\nConverged at iteration {iteration+1}")
        break

    centroids_manual = new_centroids

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

stored_iterations = []
centroids_temp = X_scaled[random_indices]

for iteration in range(max_iterations):

    distances = np.linalg.norm(X_scaled[:, np.newaxis] - centroids_temp, axis=2)
    labels_temp = np.argmin(distances, axis=1)

    new_centroids = np.array([
        X_scaled[labels_temp == i].mean(axis=0)
        for i in range(k)
    ])

    sse = 0
    for i in range(k):
        cluster_points = X_scaled[labels_temp == i]
        sse += np.sum((cluster_points - new_centroids[i])**2)

    stored_iterations.append((iteration+1, labels_temp.copy(), new_centroids.copy(), sse))

    if np.all(np.abs(new_centroids - centroids_temp) < tolerance):
        break

    centroids_temp = new_centroids


selected_steps = [
    stored_iterations[0],
    stored_iterations[len(stored_iterations)//2],
    stored_iterations[-1]
]

for ax, (step, labels_plot, centroids_plot, sse) in zip(axes, selected_steps):
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_plot)
    ax.scatter(centroids_plot[:, 0], centroids_plot[:, 1], marker='+', s=250)
    ax.set_title(f"Iteration {step}\nSSE = {round(sse, 2)}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.suptitle("Manual K-Means Clustering (k = 3)")
plt.tight_layout()
plt.show()


k_final = 10
kmeans_10 = KMeans(n_clusters=k_final, random_state=42, n_init=10)
df["Cluster_10"] = kmeans_10.fit_predict(X_scaled)


#pie chart
cluster_counts = df["Cluster_10"].value_counts().sort_index()

plt.figure(figsize=(6,6))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
plt.title("Cluster Distribution (k = 10)")
plt.show()

total_points = len(df)

print("\n==============================")
print("PIE CHART DATA (FULL VALUES)")
print("==============================")

for cluster_id, count in cluster_counts.items():
    percentage = (count / total_points) * 100
    print(f"Cluster {cluster_id}:")
    print(f"  Count      = {count}")
    print(f"  Percentage = {round(percentage,2)} %\n")