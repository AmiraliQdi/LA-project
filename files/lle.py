import numpy as np
import matplotlib.pyplot as plt
from dataset import load_dataset
import numpy as np
from geo import KNearestNeighbors

class LLE:
    """
    Locally Linear Embedding for nonlinear dimensionality reduction.
    """
    
    def __init__(self, n_components, *, adj_calculator=KNearestNeighbors(5)):
        """
        Initialize LLE with the number of components and neighbors.

        Parameters:
        - n_components: int, the number of dimensions to retain in the reduced space.
        - adj_calculator: function, given a dataset, returns the adjacency matrix.
        """
        self.n_components = n_components
        self._adj_calculator = adj_calculator
        
    def _compute_weights(self, X, indices):
        """
        Compute reconstruction weights for each data point in LLE.

        Parameters:
        - X: numpy array, shape (n_samples, n_features).
        - indices: numpy array, shape (n_samples, k), containing indices of k-nearest neighbors for each sample.

        Returns:
        - W: (n_samples, n_samples) sparse weight matrix.
        """
        n_samples, k = indices.shape
        W = np.zeros((n_samples, n_samples))  # Full (n_samples, n_samples) weight matrix

        for i in range(n_samples):
            neighbors = X[indices[i]]  # Get k-nearest neighbors of point i
            Xi = X[i]

            # Compute local covariance matrix
            G = neighbors - Xi  # Center neighbors
            C = G @ G.T  # Compute covariance matrix (k, k)

            # Regularization for numerical stability
            C += np.eye(k) * 1e-3 * np.trace(C)

            # Solve for weights
            w = np.linalg.solve(C, np.ones(k))
            w /= np.sum(w) 

            # Store weights in the correct positions of W
            W[i, indices[i]] = w  

        return W

    def _compute_embedding(self, W):
        """
        Compute the low-dimensional embedding.

        Parameters:
        - W: (n_samples, n_samples) weight matrix.

        Returns:
        - (n_samples, n_components) low-dimensional embedding.
        """
        n_samples = W.shape[0]
        
        # Step 1: Compute the matrix M = (I - W)^T * (I - W)
        I = np.eye(n_samples)  
        M = (I - W).T @ (I - W) 

        # Step 2: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(M)  # Use eigh (symmetric matrix)

        # Step 3: Sort eigenvalues in ascending order and ignore the first (zero) eigenvalue
        sorted_indices = np.argsort(eigenvalues)
        eigenvectors_sorted = eigenvectors[:, sorted_indices[1:]] 

        return eigenvectors_sorted


    def fit_transform(self, X):
        """
        Fit the LLE model to the dataset and reduce its dimensionality.

        Parameters:
        - X: numpy array, the dataset (m x n).
        """
        # Step 1: Find nearest neighbors using the custom KNearestNeighbors class
        adjacency_matrix = self._adj_calculator(X)  # Compute adjacency matrix
        indices = np.argsort(-adjacency_matrix, axis=1)[:, :self._adj_calculator.k]  # Get k-nearest indices
        
        # Step 2: Compute reconstruction weights
        W = self._compute_weights(X, indices)
        
        # Step 3: Compute the low-dimensional embedding
        embedding = self._compute_embedding(W)[:, :self.n_components]  # Take first n_components eigenvectors
        
        return embedding



if __name__ == "__main__":
    X, color = load_dataset('./files/datasets/swissroll.npz')

    # Step 2: Perform LLE
    lle = LLE(n_components=2, adj_calculator=KNearestNeighbors(10))  # Using 10 nearest neighbors
    X_reduced = lle.fit_transform(X)

    # Step 3: Visualize the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.colorbar(label="Position in Swiss Roll")
    plt.title("Swiss Roll Unfolded Using LLE")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
