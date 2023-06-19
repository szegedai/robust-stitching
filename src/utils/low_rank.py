import numpy as np
import torch
from matplotlib import pyplot as plt

from src.models.frank import FrankeinsteinNet


def low_rank_approx(A: np.ndarray, rank) -> np.ndarray:
    U, S, Vt = np.linalg.svd(A)
    S[rank:] = 0.0
    return (U * S) @ Vt


def get_transform_singular_values(model: FrankeinsteinNet) -> np.array:
    trans_mtx_shape = model.transform.shape

    trans_mtx = model.transform.transform.weight.reshape(trans_mtx_shape)
    trans_mtx = trans_mtx.detach().cpu().numpy()

    _, S, _ = np.linalg.svd(trans_mtx)

    return S

def get_transform_rank(model: FrankeinsteinNet) -> int:
    trans_mtx_shape = model.transform.shape

    trans_mtx = model.transform.transform.weight.reshape(trans_mtx_shape)
    trans_mtx = trans_mtx.detach().cpu().numpy()

    return np.linalg.matrix_rank(trans_mtx)


def reduce_transform_rank(model: FrankeinsteinNet, rank: int) -> None:    
    device = model.device
    trans_mtx_shape = model.transform.shape
    orig_shape = model.transform.transform.weight.shape

    trans_mtx = model.transform.transform.weight.reshape(trans_mtx_shape)
    trans_mtx = trans_mtx.detach().cpu().numpy()

    low_rank_trans_mtx = low_rank_approx(trans_mtx, rank).reshape(orig_shape)
    low_rank_trans_mtx = torch.tensor(data=low_rank_trans_mtx, device=device)
    low_rank_trans_mtx = torch.nn.Parameter(low_rank_trans_mtx)

    model.transform.transform.weight = low_rank_trans_mtx
