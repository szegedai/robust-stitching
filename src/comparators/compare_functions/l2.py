import numpy as np


def l2(x1, x2):
    try:
        distance = np.mean((x1-x2)**2)
    except:
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        distance = np.mean((x1-x2)**2)

    return distance
