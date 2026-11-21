# Modified from EvoPress (https://github.com/IST-DASLab/EvoPress), src/linalg_utils.py, licensed under the Apache License 2.0.
# Changes: none.
import torch
from torch import Tensor


__all__ = ["inv_sym"]


def inv_sym(X: Tensor):
    """
    More efficient and stable inversion of symmetric matrices.
    """
    return torch.cholesky_inverse(torch.linalg.cholesky(X))
