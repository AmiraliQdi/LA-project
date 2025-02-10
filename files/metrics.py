import numpy as np

def trustworthiness(D, D_embedded, *, n_neighbors=10):
    """
    Computes the trustworthiness score to evaluate how well the local structure 
    is preserved after dimensionality reduction.

    Parameters:
    - D: numpy array (n, n), distance matrix in the original high-dimensional space.
    - D_embedded: numpy array (n, n), distance matrix in the lower-dimensional space.
    - n_neighbors: int, the number of nearest neighbors to consider.

    Returns:
    - float: Trustworthiness score in the range [0, 1], where 1 indicates perfect preservation.
    """
    n = D.shape[0]  # Number of points

    # Find k-nearest neighbors in the original space (excluding self)
    orig_neighbors = np.argsort(D, axis=1)[:, 1:n_neighbors+1]  

    # Compute rank matrix in the embedded space
    rank_matrix = np.argsort(np.argsort(D_embedded, axis=1), axis=1)  

    # Get ranks of k-nearest neighbors in the embedded space
    ranks_in_embedded = rank_matrix[np.arange(n)[:, None], orig_neighbors]

    # Compute penalty: max(0, (r(i, j) - k)) for all misplaced points
    penalty = np.sum(np.maximum(0, ranks_in_embedded - n_neighbors))

    # Compute final trustworthiness score
    T_k = 1 - (2 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))) * penalty

    return T_k
