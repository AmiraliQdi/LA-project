
import numpy as np


def _compute_distance_matrix(X):
    """
    Compute the pairwise Euclidean distance matrix for X in a vectorized manner.
    
    Parameters:
    - X: numpy array of shape (m, n)
    
    Returns:
    - D: numpy array of shape (m, m) where D[i, j] is the Euclidean distance between X[i] and X[j]
    """
    sum_X = np.sum(X ** 2, axis=1)
    D_squared = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * (X @ X.T)
    D_squared[D_squared < 0] = 0.0
    D = np.sqrt(D_squared)
    return D


class KNearestNeighbors:
    """
    Compute the k-nearest neighbors for each point in the dataset.
    
    Attributes:
    - k: int, the number of nearest neighbors to find.
    """
    
    def __init__(self, k):
        self.k = k
    
    def __call__(self, X):
        """
        For each point in X, find the indices of the k nearest neighbors and build an adjacency matrix.
        
        Parameters:
        - X: numpy array of shape (m, n)
        
        Returns:
        - neighbors: numpy array (m x m) where neighbors[i, j] = 1 if j is among the k nearest neighbors of i, else 0.
        """
        D = _compute_distance_matrix(X)
        m = X.shape[0]
        # Exclude self by setting the diagonal to infinity
        np.fill_diagonal(D, np.inf)
        # Get sorted indices along each row; shape (m, m)
        sorted_indices = np.argsort(D, axis=1)
        # Select the first k indices for each row (nearest neighbors)
        knn_indices = sorted_indices[:, :self.k]
        # Initialize adjacency matrix with zeros
        neighbors = np.zeros((m, m), dtype=int)
        # Use advanced indexing to set neighbors to 1
        row_indices = np.arange(m)[:, None]  # shape (m, 1)
        neighbors[row_indices, knn_indices] = 1
        return neighbors
    

class EpsNeighborhood:
    """
    Compute the epsilon-neighborhood for each point in the dataset.
    
    Attributes:
    - epsilon: float, the maximum distance to consider a point as a neighbor.
    """
    
    def __init__(self, eps):
        self.eps = eps
    
    def __call__(self, X):
        """
        For each point in X, mark as neighbors those points within the epsilon distance.
        
        Parameters:
        - X: numpy array of shape (m, n)
        
        Returns:
        - neighbors: numpy array (m x m) where neighbors[i, j] = 1 if the distance between i and j is <= epsilon, else 0.
        """
        D = _compute_distance_matrix(X)
        # Create a binary matrix: 1 where distance is within epsilon, else 0
        neighbors = (D <= self.eps).astype(int)
        # Exclude self-neighbors (set the diagonal to 0)
        np.fill_diagonal(neighbors, 0)
        return neighbors


# --- Optional Testing ---
if __name__ == "__main__":
    # Create synthetic data (10 samples, 3 dimensions)
    np.random.seed(42)
    X = np.random.rand(10, 3)
    X = np.array([[0,0],[1,1],[2,2],[3,3]])
    
    # Test _compute_distance_matrix
    D = _compute_distance_matrix(X)
    print("Distance matrix:\n", D)
    
    # Test KNearestNeighbors with k = 3
    knn = KNearestNeighbors(k=2)
    knn_neighbors = knn(X)
    print("\nKNearestNeighbors adjacency matrix (k=3):\n", knn_neighbors)
    
    # Test EpsNeighborhood with epsilon = 0.5
    eps_neigh = EpsNeighborhood(eps=1.5)
    eps_neighbors = eps_neigh(X)
    print("\nEpsNeighborhood adjacency matrix (epsilon=0.5):\n", eps_neighbors)
