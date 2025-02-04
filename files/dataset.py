import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_dataset(path):
    dataset = np.load(path)
    return dataset['data'], dataset['target']

def generate_plane(d=2, dim=3, classes=2, num_points=1000, noise=0.1):
    """Generate a noisy d-dimensional plane within a dim-dimensional space partitioned into classes."""
    # Generate random d-dimensional points
    X = np.random.uniform(-1, 1, (num_points, d))
    
    # Extend to higher dimensions by adding zero-padding
    X_high_dim = np.hstack([X, np.zeros((num_points, dim - d))])
    
    # Add Gaussian noise
    X_high_dim += np.random.normal(0, noise, X_high_dim.shape)
    
    # Labeling strategy: Partition the hyperplane into classes
    labels = np.zeros(num_points, dtype=int)
    if classes > 1:
        # Example: Partition into a grid in the first two dimensions
        labels = ((X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)) % classes
    
    return X_high_dim, labels

if __name__ == "__main__":
    from pca import PCA  # Ensure PCA class is available
    
    # Generate and visualize the hyperplane
    X, y = generate_plane(d=2, dim=3, classes=4)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
    ax.set_title('Generated 2D Hyperplane in 3D Space')
    plt.show()
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    rec_pca = pca.inverse_transform(X_pca)
        
    # Visualize the projected data
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection of the Hyperplane')
    plt.colorbar()
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rec_pca[:, 0], rec_pca[:, 1], rec_pca[:, 2], c=y, cmap='viridis')
    ax.set_title('Reconstruced Hyperplane in 3D Space')
    plt.show()


