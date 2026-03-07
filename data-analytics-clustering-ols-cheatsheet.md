
### 1. Elbow Method (for K-Means $k$ selection)

Used to find the optimal number of clusters by plotting the Within-Cluster Sum of Squares (WCSS or Inertia) against $k$.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

k_range = range(1, 11)
wcss = []

for k in k_range:
    # Always specify n_init and random_state for consistency
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) # inertia_ stores the WCSS

# Plotting
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS / Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

```

### 2. Silhouette Score Plot (for K-Means $k$ selection)

Evaluates cluster quality. Unlike the Elbow method, you want to maximize this score. Note: Silhouette score cannot be computed for $k=1$.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_range = range(2, 11) # Must start at 2
sil_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)

# Plotting
plt.plot(k_range, sil_scores, marker='s')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.grid(True)
plt.show()

```

### 3. K-Distance Graph (for DBSCAN $\epsilon$ selection)

Used to find the optimal `eps` parameter for DBSCAN by plotting the distance to the $k$-th nearest neighbor. The "elbow" or steep rise in this plot indicates the ideal `eps`.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

k = 5 # This usually corresponds to your chosen min_samples
# Fit NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, indices = nbrs.kneighbors(X)

# Sort distances of the k-th neighbor (index k-1)
distances_sorted = np.sort(distances[:, k-1], axis=0)

# Plotting
plt.plot(distances_sorted)
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.xlabel('Points sorted by distance')
plt.title(f'K-Distance Graph (k={k})')
plt.grid(True)
plt.show()

```

### 4. Dendrogram (for Hierarchical Clustering)

Visualizes the hierarchical relationships and helps determine the number of clusters by cutting the tree.

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Calculate linkage matrix (methods: 'ward', 'single', 'complete', 'average')
linkage_matrix = linkage(X, method='ward')

# Plotting
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data points')
plt.ylabel('Distance')
plt.show()

```

---

## Quick Reference for Documentation Lookups

You'll be using the docs for the exact syntax of these algorithms, but here is a quick map of where to look and what to remember:

### Optimization (`scipy.optimize`)

* **Univariate:** `minimize_scalar(func, bounds=(low, high), method='bounded')`. Remember to return $-f(x)$ if maximizing.
* **Multivariate:** `minimize(func, x0, method='BFGS')`. Requires an initial guess `x0`.
* **Linear Programming (LP):** `linprog(c, A_ub, b_ub, bounds)`. *Warning:* SciPy minimizes $c^T x$. If maximizing profit, negate your $c$ array when passing it in.
* **Quadratic Programming (QP):** Solved using `minimize` with `method='SLSQP'` and passing `constraints` and `bounds` dictionaries/lists.

### PCA (`sklearn.decomposition`)

* **Crucial Pre-step:** You *must* standardize your data before applying PCA to prevent features with larger scales from dominating the variance.
* **Standardization:** `StandardScaler().fit_transform(X)`.
* **Variance:** Extract the explained variance using `pca.explained_variance_ratio_`.

### Clustering Quality Metrics

* **Scikit-Learn native:** `silhouette_score(X, labels)` and `adjusted_rand_score(labels_true, labels_pred)`.
* **Custom implementations:** Purity and Entropy require custom calculations using contingency matrices or `scipy.stats.entropy`, as they are not native single-line sklearn metrics.

### Regression (`sklearn.linear_model`)

* **Library approach:** `LinearRegression().fit(X, y)`. Use `model.intercept_` for $w_0$ and `model.coef_` for slopes.
* **OLS Manual Math:** If asked to use the normal equation: $\hat{w} = (X^T X)^{-1} X^T y$. In NumPy, you must add a column of ones to $X$ for the bias term first (`np.hstack([np.ones((n, 1)), X])`), then compute using `@` for matrix multiplication and `numpy.linalg.inv`.
