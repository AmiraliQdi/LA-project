import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import load_dataset

class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    """

    def __init__(self, n_components):
        """
        Initialize PCA with the number of components to retain.

        Parameters:
        - n_components: int, the number of principal components to keep.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None

    def _center_data(self, X):
        """Compute the mean of X along axis 0 (features) and subtract it from X."""
        self.mean = np.mean(X, axis=0)
        return X - self.mean

    def _create_cov(self, X):
        """Compute the covariance matrix of X."""
        return np.cov(X, rowvar=False)

    def _decompose(self, covariance_matrix):
        """Perform eigendecomposition on the covariance matrix."""
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

    def fit(self, X):
        """Fit the PCA model to the dataset by computing the principal components."""
        X_centered = self._center_data(X)
        covariance_matrix = self._create_cov(X_centered)
        eigenvalues, eigenvectors = self._decompose(covariance_matrix)
        self.components = eigenvectors[:, :self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
    
    def transform(self, X):
        """Project the data onto the top principal components."""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """Fit the PCA model and transform the data in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Reconstruct the original data from the transformed data."""
        return np.dot(X_transformed, self.components.T) + self.mean


if __name__ == "__main__":
    
    X, y = load_dataset('./files/datasets/swissroll.npz')
    
    # Apply PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Visualize the reduced data
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection of Swiss Roll')
    plt.colorbar()
    plt.show()
    
    # Reconstruct the original dataset
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Visualize the original and reconstructed Swiss Roll side by side
    fig = plt.figure(figsize=(12, 6))
    
    # Original Swiss Roll
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
    ax1.set_title('Original Swiss Roll')
    
    # Reconstructed Swiss Roll
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], X_reconstructed[:, 2], c=y, cmap='viridis')
    ax2.set_title('Reconstructed Swiss Roll')
    
    plt.show()
