import numpy as np
from util import plot_explained_variance, plot_images
from dataset import load_dataset
from pca import PCA

if __name__ == "__main__":
    data, target = load_dataset("./files/datasets/faces.npz")
    
    # Number of components to test
    n_components_list = [5, 20, 50, 100]
    
    for n_components in n_components_list:
        
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
        reconstructed_data = pca.inverse_transform(transformed_data)

        # Visualize results with explained variance ratio
        explained_variance = np.sum(pca.explained_variance_ratio_)
        plot_images(data, reconstructed_data, n_components,explained_variance=explained_variance)

        # Visualize explained variance ratio
    pca_full = PCA(n_components=data.shape[1])
    pca_full.fit(data)
    plot_explained_variance(pca_full)
    
    # Print explained variance ratio
    print("Explained Variance Ratio:", pca_full.explained_variance_ratio_)
        
