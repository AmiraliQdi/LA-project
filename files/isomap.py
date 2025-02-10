import numpy as np
import networkx as nx
from geo import KNearestNeighbors
from pca import PCA
# from util import dijkstra
from scipy.sparse.csgraph import dijkstra
from dataset import load_dataset
import matplotlib.pyplot as plt

class Isomap:
    """
    Isomap for dimensionality reduction by preserving geodesic distances.
    """

    def __init__(self, n_components, *, adj_calculator=KNearestNeighbors(5), decomposer=None):
        """
        Initialize Isomap with the number of components and neighbors.

        Parameters:
        - n_components: int, the number of dimensions to retain in the reduced space.
        - adj_calculator: function, given a dataset, returns the adjacency matrix.
        """
        self.n_components = n_components
        self._adj_calculator = adj_calculator
        self._decomposer = decomposer or PCA(n_components=n_components)

    def _compute_geodesic_distances(self, X):
        """
        Compute the geodesic distance matrix using Dijkstra's algorithm.
        """
        adjacency_matrix = self._adj_calculator(X)
        geodesic_distances = dijkstra(adjacency_matrix, directed=False)
        return geodesic_distances

    def _decompose(self, geodesic_distances):
        """
        Apply MDS (eigen-decomposition) to the geodesic distance matrix.
        """
        n = geodesic_distances.shape[0]
        I = np.eye(n)
        J = np.ones((n, n)) / n
        C = I - J
        
        D_squared = np.square(geodesic_distances)
        B = -0.5 * C @ D_squared @ C

        return self._decomposer.fit_transform(B)

    def fit_transform(self, X):
        """
        Fit the Isomap model to the dataset and reduce its dimensionality.

        Parameters:
        - X: numpy array, the dataset (m x n).
        """
        geodesic_distances = self._compute_geodesic_distances(X)
        X_transformed = self._decompose(geodesic_distances)
        return X_transformed

if __name__ == "__main__":

    X, y = load_dataset('./files/datasets/swissroll.npz')
    
    # Apply Isomap
    isomap = Isomap(n_components=2)
    X_isomap = isomap.fit_transform(X)
    
    # Visualize the reduced data
    plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y, cmap='viridis')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Isomap Projection of Swiss Roll')
    plt.colorbar()
    plt.show()

