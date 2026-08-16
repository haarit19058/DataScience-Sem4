import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# ==========================================
# 1. Spectral Clustering
# ==========================================
class SpectralClustering:
    def __init__(self, n_clusters=2, random_state=42):
        self.k = n_clusters
        self.rng = np.random.default_rng(random_state)

    def _kmeans_scratch(self, X, max_iter=100):
        n, d = X.shape
        centers = X[self.rng.choice(n, self.k, replace=False)]
        labels = np.zeros(n, dtype=int)
        
        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for j in range(self.k):
                if np.sum(labels == j) > 0:
                    centers[j] = X[labels == j].mean(axis=0)
        return labels

    def fit_predict(self, adj_matrix):
        degrees = np.sum(adj_matrix, axis=1)
        d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
        D_inv_sqrt_mat = np.diag(d_inv_sqrt)
        
        L_sym = np.eye(len(degrees)) - D_inv_sqrt_mat @ adj_matrix @ D_inv_sqrt_mat
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        U = eigenvectors[:, :self.k]
        row_norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
        T = U / row_norms
        
        return self._kmeans_scratch(T)

# ==========================================
# 2. Charikar's Greedy Densest Subgraph
# ==========================================
def charikars_densest_subgraph(adj_matrix):
    n = adj_matrix.shape[0]
    A = adj_matrix.copy()
    active_nodes = set(range(n))
    degrees = np.sum(A, axis=1).astype(float)

    density_history = []
    subsets_history = []

    while active_nodes:
        current_edges = np.sum(A) / 2.0
        current_density = current_edges / len(active_nodes)
        
        density_history.append(current_density)
        subsets_history.append(set(active_nodes))

        active_list = list(active_nodes)
        min_node = min(active_list, key=lambda u: degrees[u])
        active_nodes.remove(min_node)

        for neighbor in active_nodes:
            if A[min_node, neighbor] > 0:
                degrees[neighbor] -= A[min_node, neighbor]
        
        A[min_node, :] = 0
        A[:, min_node] = 0
        degrees[min_node] = 0

    best_idx = int(np.argmax(density_history))
    return subsets_history[best_idx], density_history[best_idx], density_history

# ==========================================
# 3. Flow-based Community / Min s-t Cut
# ==========================================
class EdmondsKarpMinCut:
    def __init__(self, adj_matrix):
        self.capacity = adj_matrix.copy().astype(float)
        self.n = adj_matrix.shape[0]

    def _bfs(self, s, t, parent):
        visited = np.zeros(self.n, dtype=bool)
        queue = deque([s])
        visited[s] = True
        
        while queue:
            u = queue.popleft()
            for v in range(self.n):
                if not visited[v] and self.residual[u, v] > 1e-9:
                    visited[v] = True
                    parent[v] = u
                    if v == t:
                        return True
                    queue.append(v)
        return False

    def compute_min_cut(self, source, sink):
        self.residual = self.capacity.copy()
        parent = np.full(self.n, -1, dtype=int)
        max_flow = 0.0

        while self._bfs(source, sink, parent):
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.residual[parent[s], s])
                s = parent[s]
            
            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                self.residual[u, v] -= path_flow
                self.residual[v, u] += path_flow
                v = parent[v]

        visited = np.zeros(self.n, dtype=bool)
        queue = deque([source])
        visited[source] = True
        while queue:
            u = queue.popleft()
            for v in range(self.n):
                if not visited[v] and self.residual[u, v] > 1e-9:
                    visited[v] = True
                    queue.append(v)
                    
        return np.where(visited)[0], np.where(~visited)[0], max_flow

# ==========================================
# Data Generation & Benchmarking
# ==========================================
def generate_stochastic_block_model(n_nodes=60, p_in=0.55, p_out=0.03, random_state=42):
    rng = np.random.default_rng(random_state)
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    half = n_nodes // 2
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            same_comm = (i < half and j < half) or (i >= half and j >= half)
            if rng.random() < (p_in if same_comm else p_out):
                adj[i, j] = adj[j, i] = 1.0
    return adj

def plot_empirical_complexity():
    sizes = [20, 40, 60, 80, 100, 120, 150]
    t_spectral, t_charikar, t_flow = [], [], []

    print("\nBenchmarking Algorithms (this takes a few seconds)...")
    for n in sizes:
        adj = generate_stochastic_block_model(n_nodes=n, p_in=0.6, p_out=0.05, random_state=n)
        
        start = time.perf_counter()
        SpectralClustering(n_clusters=2).fit_predict(adj)
        t_spectral.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        charikars_densest_subgraph(adj)
        t_charikar.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        EdmondsKarpMinCut(adj).compute_min_cut(source=0, sink=n-1)
        t_flow.append(time.perf_counter() - start)

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, t_spectral, marker='o', label='Spectral (Matrix Eigendecomp)', color='blue')
    plt.plot(sizes, t_charikar, marker='s', label="Charikar's Peeling (Array ops)", color='crimson')
    plt.plot(sizes, t_flow, marker='^', label='Edmonds-Karp Min-Cut (BFS)', color='green')
    
    plt.title("Empirical Time Complexity Scaling", fontweight='bold')
    plt.xlabel("Number of Nodes (V)")
    plt.ylabel("Execution Time (Seconds)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# Visualizing Results
# ==========================================
def visualize_community_results(adj, spec_labels, density_history, cut_S):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sorted_indices = np.argsort(spec_labels)
    sorted_adj = adj[sorted_indices, :][:, sorted_indices]
    axes[0].imshow(sorted_adj, cmap='Blues', interpolation='nearest')
    axes[0].set_title("Spectral: Block Adjacency Matrix", fontweight='bold')
    
    axes[1].plot(range(len(density_history)), density_history, color='crimson', lw=2)
    max_step = int(np.argmax(density_history))
    axes[1].axvline(max_step, color='black', linestyle='--', label=f'Max Density Step ({max_step})')
    axes[1].scatter(max_step, density_history[max_step], color='gold', s=100, zorder=5, edgecolors='black')
    axes[1].set_title("Charikar's Peeling: Density Curve", fontweight='bold')
    axes[1].legend()

    n = adj.shape[0]
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.column_stack([np.cos(theta), np.sin(theta)])
    
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                is_cut = (i in cut_S and j not in cut_S) or (i not in cut_S and j in cut_S)
                axes[2].plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                             color='red' if is_cut else 'lightgray', alpha=0.8 if is_cut else 0.25, lw=1)

    colors = ['#1f77b4' if i in cut_S else '#2ca02c' for i in range(n)]
    axes[2].scatter(pos[:, 0], pos[:, 1], c=colors, s=80, edgecolors='black', zorder=5)
    axes[2].set_title("Flow Min-Cut (Red = Cut Edges)", fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_nodes = 60
    adj_matrix = generate_stochastic_block_model(n_nodes=n_nodes, p_in=0.55, p_out=0.03, random_state=42)
    
    # 1. Spectral
    spec_labels = SpectralClustering(n_clusters=2).fit_predict(adj_matrix)
    
    # 2. Charikar
    dense_sub, max_den, den_curve = charikars_densest_subgraph(adj_matrix)
    
    # 3. Flow Min-Cut
    comm_S, comm_T, flow_val = EdmondsKarpMinCut(adj_matrix).compute_min_cut(source=0, sink=59)

    # --- PRINT FINAL ANSWERS ---
    print("="*50)
    print("1. SPECTRAL CLUSTERING RESULTS")
    print("="*50)
    print(f"Cluster 0 contains nodes:\n{np.where(spec_labels == 0)[0]}")
    print(f"Cluster 1 contains nodes:\n{np.where(spec_labels == 1)[0]}")
    
    print("\n" + "="*50)
    print("2. CHARIKAR'S DENSEST SUBGRAPH RESULTS")
    print("="*50)
    print(f"Densest subset size: {len(dense_sub)} nodes")
    print(f"Maximum Density (|E|/|V|): {max_den:.4f}")
    print(f"Nodes in Densest Subgraph:\n{sorted(list(dense_sub))}")
    
    print("\n" + "="*50)
    print("3. FLOW-BASED MIN-CUT (Source=0, Sink=59)")
    print("="*50)
    print(f"Source Community (S) Size: {len(comm_S)}")
    print(f"Sink Community (T) Size: {len(comm_T)}")
    print(f"Max Flow (Number of Cut Edges): {int(flow_val)}")
    print(f"Nodes in Source Community S:\n{comm_S}")
    print("="*50 + "\n")

    # --- VISUALIZATIONS ---
    visualize_community_results(adj_matrix, spec_labels, den_curve, comm_S)
    plot_empirical_complexity()
