from sklearn.metrics import r2_score
import numpy as np

def rearrange_activations(activations):
    batch_size = activations.shape[0]
    flat_activations = activations.reshape(batch_size, -1)
    return flat_activations

def r2(x1, x2):
    x1_flat, x2_flat = rearrange_activations(x1), rearrange_activations(x2)
    return r2_score(x1_flat, x2_flat)