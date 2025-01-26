import numpy as np


def generate_plane(d=2, dim=3, classes=2):
    # TODO: generate a noisy d-dimensional plane within
    # a dim-dimensional space partitioned into classes
    return None, None


def load_dataset(path):
    dataset = np.load(path)
    return dataset['data'], dataset['target']
