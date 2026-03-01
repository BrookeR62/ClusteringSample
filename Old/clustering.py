# =========================
# STEP 1: IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# =========================
# STEP 2: LOAD DATASET
# =========================

print("Loading dataset...")

df = pd.read_csv("studentdata_ordinal_kmeans.csv", encoding="latin1")

df.columns = df.columns.str.strip()

data = df.select_dtypes(include=[np.number])

data = data.fillna(data.mean())

print("Dataset ready.")


# =========================
# STEP 3: ELBOW METHOD GRAPH (k vs SSE)
# =========================

print("\nGenerating Elbow Method graph...")

k_values = range(2, 11)

sse_values = []
silhouette_values = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=32,
        n_init=10
    )

    labels = model.fit_predict(data)

    sse_values.append(model.inertia_)

    sil_score = silhouette_score(data, labels)

    silhouette_values.append(sil_score)

    print(f"k={k}, SSE={model.inertia_:.2f}, Silhouette={sil_score:.4f}")


plt.figure()

plt.plot(k_values, sse_values, marker='o')

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method")

plt.savefig("elbow_method.png")

plt.show()

print("Saved: elbow_method.png")



# =========================
# STEP 4: SILHOUETTE METHOD GRAPH (k vs silhouette score)
# =========================

print("\nGenerating Silhouette Method graph...")

plt.figure()

plt.plot(k_values, silhouette_values, marker='o')

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")

plt.savefig("silhouette_method.png")

plt.show()

print("Saved: silhouette_method.png")




print("\nApplying final clustering...")

kmeans = KMeans(
    n_clusters=3,
    random_state=32,
    n_init=10
)

clusters = kmeans.fit_predict(data)

df["Cluster"] = clusters

df.to_csv("studentdata_ordinal_kmeans.csv", index=False)

print("Saved: studentdata_clustered.csv")


# =========================
# STEP 6: ELBOW CLUSTER VISUALIZATION PER k
# =========================

# =========================
# STEP 5A: ELBOW PIE CHARTS
# =========================

print("\nGenerating Elbow Method Pie Charts...")

for k in range(2, 6):

    model = KMeans(
        n_clusters=k,
        random_state=32,
        n_init=10
    )

    labels = model.fit_predict(data)

    counts = pd.Series(labels).value_counts().sort_index()

    plt.figure()

    plt.pie(
        counts,
        labels=[f"Cluster {i}" for i in counts.index],
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(f"Elbow Method Cluster Distribution (k={k})")

    filename = f"elbow_pie_k{k}.png"

    plt.savefig(filename)

    plt.show()

    print(f"Saved: {filename}")
# =========================

# =========================
# STEP 5B: SILHOUETTE PIE CHARTS
# =========================

print("\nGenerating Silhouette Method Pie Charts...")

for k in range(2, 6):

    model = KMeans(
        n_clusters=k,
        random_state=32,
        n_init=10
    )

    labels = model.fit_predict(data)

    sil = silhouette_score(data, labels)

    counts = pd.Series(labels).value_counts().sort_index()

    plt.figure()

    plt.pie(
        counts,
        labels=[f"Cluster {i}" for i in counts.index],
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(f"Silhouette Method Cluster Distribution (k={k}) Score={sil:.3f}")

    filename = f"silhouette_pie_k{k}.png"

    plt.savefig(filename)

    plt.show()

    print(f"Saved: {filename}")

# =========================
# STEP 8: COMPLETE
# =========================

print("\nPROCESS COMPLETE")
print("Generated files:")
print("elbow_method.png")
print("silhouette_method.png")
print("elbow_clusters.png")
print("silhouette_clusters.png")
print("studentdata_clustered.csv")