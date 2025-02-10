import numpy as np
import heapq
import matplotlib.pyplot as plt

def dijkstra(adjacency_matrix, directed=False):
    """
    Compute shortest paths using Dijkstra's algorithm in a memory-efficient way.

    Parameters:
    - adjacency_matrix: numpy array (m, m), graph adjacency with np.inf for non-edges.
    - directed: bool, whether the graph is directed.

    Returns:
    - geodesic_distances: numpy array (m, m) with shortest path distances.
    """
    m = adjacency_matrix.shape[0]
    geodesic_distances = np.full((m, m), np.inf)
    for start_node in range(m):
        # Priority queue (min-heap): (distance, node)
        min_heap = [(0, start_node)]
        visited = np.zeros(m, dtype=bool)
        distances = np.full(m, np.inf)
        distances[start_node] = 0

        while min_heap:
            current_distance, current_node = heapq.heappop(min_heap)

            if visited[current_node]:
                continue
            visited[current_node] = True

            # Get all neighbors (non-infinity)
            neighbors = np.where(adjacency_matrix[current_node] < np.inf)[0]
            new_distances = current_distance + adjacency_matrix[current_node, neighbors]

            # Only update if new distance is smaller
            update_mask = new_distances < distances[neighbors]
            distances[neighbors[update_mask]] = new_distances[update_mask]

            # Push updated distances to the heap
            for neighbor, new_dist in zip(neighbors[update_mask], new_distances[update_mask]):
                heapq.heappush(min_heap, (new_dist, neighbor))

        geodesic_distances[start_node] = distances

    return geodesic_distances


def plot_images(original, reconstructed, n_components,n_samples=5,explained_variance=None):
    """Plot original and reconstructed images side by side."""
    fig, axes = plt.subplots(2, n_samples, figsize=(10, 5))
    
    for i in range(n_samples):
        axes[0, i].imshow(original[i].reshape(32, 32), cmap='gray')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed[i].reshape(32, 32), cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_title("Original Images")
    axes[1, 0].set_title("Reconstructed Images")
    
    if explained_variance is not None:
        plt.figtext(0.5, 0.01, f"Explained Variance: {explained_variance:.4f} | number of components={n_components}", ha="center", fontsize=10)
    
    plt.show()

def plot_explained_variance(pca):
    """Plot cumulative explained variance ratio."""
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_,), marker='o', linestyle='--')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Explained Variance vs. Number of Components")
    plt.grid()
    plt.show()

def plot_thrustworthiness_ncomponents(thrust_array,algorithm):
    """Plot cumulative explained variance ratio."""
    plt.figure(figsize=(8, 5))
    plt.plot(thrust_array, marker='o', linestyle='--')
    plt.xlabel("Number of Components")
    plt.ylabel("Thrusworthiness score")
    plt.title(f"{algorithm}:Thrusworthiness score based on diffrent number of componentens")
    plt.grid()
    plt.show()