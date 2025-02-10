import numpy as np
import matplotlib.pyplot as plt
from pca import PCA
from isomap import Isomap
from lle import LLE
from geo import KNearestNeighbors
from dataset import load_dataset
from metrics import trustworthiness

if __name__ == "__main__":
    X, color = load_dataset('./files/datasets/swissroll.npz')

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    trust_pca = trustworthiness(np.linalg.norm(X[:, None] - X[None, :], axis=2), 
                                np.linalg.norm(X_pca[:, None] - X_pca[None, :], axis=2), 
                                n_neighbors=10)

    # Isomap
    isomap = Isomap(n_components=2)
    X_isomap = isomap.fit_transform(X)
    trust_isomap = trustworthiness(np.linalg.norm(X[:, None] - X[None, :], axis=2), 
                                   np.linalg.norm(X_isomap[:, None] - X_isomap[None, :], axis=2), 
                                   n_neighbors=10)

    # LLE
    lle = LLE(n_components=2, adj_calculator=KNearestNeighbors(10))
    X_lle = lle.fit_transform(X)
    trust_lle = trustworthiness(np.linalg.norm(X[:, None] - X[None, :], axis=2), 
                                np.linalg.norm(X_lle[:, None] - X_lle[None, :], axis=2), 
                                n_neighbors=10)

    print(f"PCA:{trust_pca} | ISO:{trust_isomap} | LLE:{trust_lle}")


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    algorithms = [("PCA", X_pca, trust_pca), 
                  ("Isomap", X_isomap, trust_isomap), 
                  ("LLE", X_lle, trust_lle)]

    for ax, (title, X_reduced, trust) in zip(axes, algorithms):
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color, cmap="viridis")
        ax.set_title(f"{title} (Trustworthiness: {trust:.3f})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.colorbar(scatter, ax=ax, label="Swiss Roll Position")

    plt.tight_layout()
    plt.show()
