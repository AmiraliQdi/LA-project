import numpy as np

import numpy as np

import numpy as np

def trustworthiness(D, D_embedded, *, n_neighbors=5):
    """
    Computes the trustworthiness score to evaluate how well the local structure 
    is preserved after dimensionality reduction.
    
    Parameters:
    - D: numpy array, the distance matrix in the original high-dimensional space.
    - D_embedded: numpy array, the distance matrix in the lower-dimensional space.
    - n_neighbors: int, the number of nearest neighbors to consider.
    
    Returns:
    - float: Trustworthiness score in the range [0, 1], where 1 indicates perfect preservation.
    """
    n = D.shape[0]  # Number of data points
    
    # Get k-nearest neighbor indices in the original and embedded spaces
    original_neighbors = np.argsort(D, axis=1)[:, 1:n_neighbors+1]
    embedded_neighbors = np.argsort(D_embedded, axis=1)[:, 1:n_neighbors+1]

    # Compute rank matrix in the embedded space
    rank_matrix = np.argsort(np.argsort(D_embedded, axis=1), axis=1)

    # Create a mask for misplaced neighbors (neighbors in original space not in embedded space)
    misplaced_mask = ~np.isin(original_neighbors, embedded_neighbors)

    # Get ranks of misplaced neighbors and compute penalty
    misplaced_ranks = rank_matrix[np.arange(n)[:, None], original_neighbors]
    penalty = np.sum(np.maximum(0, (misplaced_ranks - n_neighbors)) * misplaced_mask)

    # Compute trustworthiness score
    T_k = 1 - (2 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))) * penalty

    return T_k

