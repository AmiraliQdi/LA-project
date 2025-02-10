import numpy as np
from util import plot_thrustworthiness_ncomponents
from dataset import load_dataset
from pca import PCA
from isomap import Isomap
from lle import LLE
from metrics import trustworthiness
import matplotlib.pyplot as plt
from geo import KNearestNeighbors

if __name__ == "__main__":
    X, target = load_dataset("./files/datasets/faces.npz")
    
    n_components_list = np.arange(5,100)

    thrust_array_pca = []
    thrust_array_iso = []
    thrust_array_lle = []
    
    for n_components in n_components_list:
        
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        trust_pca = trustworthiness(np.linalg.norm(X[:, None] - X[None, :], axis=2), 
                                    np.linalg.norm(X_pca[:, None] - X_pca[None, :], axis=2), 
                                    n_neighbors=10)
        thrust_array_pca.append(trust_pca)

        # Isomap
        isomap = Isomap(n_components=n_components)
        X_isomap = isomap.fit_transform(X)
        trust_isomap = trustworthiness(np.linalg.norm(X[:, None] - X[None, :], axis=2), 
                                    np.linalg.norm(X_isomap[:, None] - X_isomap[None, :], axis=2), 
                                    n_neighbors=10)
        thrust_array_iso.append(trust_isomap)

        # LLE
        lle = LLE(n_components=n_components, adj_calculator=KNearestNeighbors(10))
        X_lle = lle.fit_transform(X)
        trust_lle = trustworthiness(np.linalg.norm(X[:, None] - X[None, :], axis=2), 
                                    np.linalg.norm(X_lle[:, None] - X_lle[None, :], axis=2), 
                                    n_neighbors=10)
        thrust_array_lle.append(trust_lle)

plot_thrustworthiness_ncomponents(thrust_array_pca,'PCA')
plot_thrustworthiness_ncomponents(thrust_array_iso,'ISO')
plot_thrustworthiness_ncomponents(thrust_array_lle,'LLE')


        
