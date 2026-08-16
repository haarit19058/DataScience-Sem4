import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd # For formatting the final table

# ==========================================
# 1. CUSTOM GAUSSIAN & DATASET GENERATION
# ==========================================

def gaussian_pdf(x, mu, sigma):
    """Returns the probability density of a Gaussian distribution at point x."""
    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coefficient * exponent

def get_random_location(n_clusters, d=2):
    """Generates random cluster centers, spaced out by a factor of 15."""
    return np.random.rand(n_clusters, d) * 15

def get_clusterable_dataset(n_points, n_clusters, d=2, sigma=1.0):
    """Generates a dataset using rejection sampling based on our custom Gaussian PDF."""
    centers = get_random_location(n_clusters, d)
    X, y_true = [], []
    points_per_cluster = n_points // n_clusters
    
    for i, center in enumerate(centers):
        n_curr_points = n_points - (points_per_cluster * (n_clusters - 1)) if i == n_clusters - 1 else points_per_cluster
        
        cluster_points = []
        max_pdf = gaussian_pdf(center, mu=center, sigma=sigma) 
        
        while len(cluster_points) < n_curr_points:
            candidate_x = np.random.uniform(low=center - 5*sigma, high=center + 5*sigma, size=(d,))
            random_prob = np.random.uniform(low=0, high=max_pdf, size=(d,))
            actual_pdf = gaussian_pdf(candidate_x, mu=center, sigma=sigma)
            
            if np.all(random_prob < actual_pdf):
                cluster_points.append(candidate_x)
                
        X.append(np.array(cluster_points))
        y_true.extend([i] * n_curr_points)
        
    return np.vstack(X), np.array(y_true)

# ==========================================
# 2. CLUSTERING ALGORITHMS & METRICS
# ==========================================

def compute_wcss(X, labels):
    """Computes the Within-Cluster Sum of Squares (Loss) for given labels."""
    wcss = 0
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            center = cluster_points.mean(axis=0)
            wcss += np.sum((cluster_points - center) ** 2)
    return wcss

def run_kmeans(X, n_clusters, initial_centers, max_iter=100):
    """Standard K-Means loop."""
    centers = np.copy(initial_centers)
    loss_history = []
    
    for i in range(max_iter):
        # 1. Assign to closest center
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 2. Record Loss
        loss = compute_wcss(X, labels)
        loss_history.append(loss)
        
        # 3. Update centers
        new_centers = np.array([X[labels == k].mean(axis=0) if len(X[labels == k]) > 0 else centers[k] for k in range(n_clusters)])
        
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
        
    return labels, loss_history

def init_kmeans_random(X, n_clusters):
    """Random initialization for standard K-Means."""
    random_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    return X[random_indices]

def init_kmeans_plusplus(X, n_clusters):
    """Probabilistic initialization for K-Means++."""
    # 1. Pick first center randomly
    centers = [X[np.random.randint(X.shape[0])]]
    
    for _ in range(1, n_clusters):
        # 2. Calculate squared distance from each point to its nearest existing center
        distances = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        
        # 3. Compute probabilities (proportional to squared distance)
        probs = distances / np.sum(distances)
        
        # 4. Choose next center based on these probabilities
        next_idx = np.random.choice(X.shape[0], p=probs)
        centers.append(X[next_idx])
        
    return np.array(centers)

# ==========================================
# 3. EXECUTION & VISUALIZATION
# ==========================================

# Generate Data (6 clusters, 600 points)
n_clusters = 6
X, y_true = get_clusterable_dataset(n_points=600, n_clusters=n_clusters, sigma=1.2)

# --- Run Agglomerative Clustering ---
Z_min = linkage(X, method='single')
Z_max = linkage(X, method='complete')
Z_avg = linkage(X, method='average')

labels_agg_min = fcluster(Z_min, n_clusters, criterion='maxclust')
labels_agg_max = fcluster(Z_max, n_clusters, criterion='maxclust')
labels_agg_avg = fcluster(Z_avg, n_clusters, criterion='maxclust')

# --- Run K-Means & K-Means++ ---
init_random = init_kmeans_random(X, n_clusters)
labels_kmeans, loss_kmeans = run_kmeans(X, n_clusters, init_random)

init_pp = init_kmeans_plusplus(X, n_clusters)
labels_kmeans_pp, loss_kmeans_pp = run_kmeans(X, n_clusters, init_pp)

# --- Calculate Final Losses ---
final_losses = {
    "Agglomerative (Single)": compute_wcss(X, labels_agg_min),
    "Agglomerative (Complete)": compute_wcss(X, labels_agg_max),
    "Agglomerative (Average)": compute_wcss(X, labels_agg_avg),
    "K-Means (Random Init)": loss_kmeans[-1],
    "K-Means++": loss_kmeans_pp[-1]
}

# --- Plot 1: Clustered Points Comparison ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Clustering Algorithms Comparison (K=6)", fontsize=18)

plot_data = [
    ("True Labels\n(Generated Data)", y_true, axes[0,0]),
    ("Agglomerative\n(Single/Min Linkage)", labels_agg_min, axes[0,1]),
    ("Agglomerative\n(Complete/Max Linkage)", labels_agg_max, axes[0,2]),
    ("Agglomerative\n(Average Linkage)", labels_agg_avg, axes[1,0]),
    ("Standard K-Means\n(Random Init)", labels_kmeans, axes[1,1]),
    ("K-Means++\n(Probabilistic Init)", labels_kmeans_pp, axes[1,2])
]

for title, labels, ax in plot_data:
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', edgecolors='k', s=40, alpha=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# --- Plot 2: Iterative Loss (K-Means vs K-Means++) ---
plt.figure(figsize=(10, 5))
plt.plot(loss_kmeans, marker='o', linestyle='-', label='Standard K-Means')
plt.plot(loss_kmeans_pp, marker='s', linestyle='-', label='K-Means++')
plt.title("Loss (WCSS) over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- Print Final Metrics Table ---
print("\n" + "="*50)
print(f"{'Clustering Algorithm':<30} | {'Final Loss (WCSS)':<15}")
print("="*50)
for algo, loss in final_losses.items():
    print(f"{algo:<30} | {loss:,.2f}")
print("="*50 + "\n")
