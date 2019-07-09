import numpy as np

def flatten_weight(weights):
    """ Flatten the weight matrix """
    weights_vec = np.array([])
    for weight in weights:
        weights_vec = np.concatenate([weights_vec, weight.flatten()])
    return weights_vec.flatten()
