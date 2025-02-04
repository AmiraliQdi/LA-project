import numpy as np

def _compute_distance_matrix(X):
    """
    Compute pairwise Euclidean distance matrix for X.
    
    Parameters:
    - X: numpy array of shape (m, n), where m is the number of samples.
    
    Returns:
    - distance_matrix: numpy array of shape (m, m) with pairwise distances.
    """
    m = X.shape[0]
    distance_matrix = np.zeros((m, m))
    
    for i in range(m):
        for j in range(m):
            distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
    
    return distance_matrix


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
        Parameters:
        - X: numpy array, the dataset (m x n).
        
        Returns:
        - neighbors: numpy array, adjacency matrix (m x m).
        """
        distance_matrix = _compute_distance_matrix(X)
        m = X.shape[0]
        adjacency_matrix = np.zeros((m, m))
        
        for i in range(m):
            # Get indices of k nearest neighbors (excluding the point itself)
            knn_indices = np.argsort(distance_matrix[i])[1:self.k + 1]
            adjacency_matrix[i, knn_indices] = distance_matrix[i, knn_indices]
            adjacency_matrix[knn_indices, i] = distance_matrix[i, knn_indices]  # Symmetric adjacency
        
        return adjacency_matrix
    

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
        Parameters:
        - X: numpy array, the dataset (m x n).
        
        Returns:
        - neighbors: numpy array, adjacency matrix (m x m).
        """
        distance_matrix = _compute_distance_matrix(X)
        adjacency_matrix = (distance_matrix <= self.eps).astype(float)
        adjacency_matrix[adjacency_matrix == 0] = np.inf  # Non-neighbors set to infinity
        np.fill_diagonal(adjacency_matrix, 0)  # Ensure zero self-distances
        
        return adjacency_matrix