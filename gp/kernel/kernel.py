from abc import ABC
from typing import List

import numpy as np


class Kernel(ABC):
    """
    Base class for all kernels
    """

    def __init__(self, nparams: int) -> None:
        self._nparams = nparams

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if x1.shape[1] != x2.shape[1]:
            raise ValueError(f"Vectors must be of matching dimension, "
                             f"but x1.shape = {x1.shape} and x2.shape = {x2.shape}")

    @staticmethod
    def _check_input_is_valid(x1: np.ndarray, x2: np.ndarray) -> None:
        if x1.shape[1] != x2.shape[1]:
            raise ValueError(f"Vectors must be of matching dimension, "
                             f"but x1.shape = {x1.shape} and x2.shape = {x2.shape}")

    def set_params(self, params: np.ndarray) -> None:
        if params.size != self.nparams:
            raise ValueError(f"Provided {params.size} params; must be {self.nparams}")

    def get_params(self) -> np.ndarray:
        raise NotImplementedError

    def get_true_params(self) -> np.ndarray:
        raise NotImplementedError

    def gradients(self, x) -> List[np.ndarray]:
        raise NotImplementedError

    def gradients_wrt_data(self, x: np.ndarray, n: int = None, dim: int = None):
        raise NotImplementedError

    @property
    def nparams(self) -> int:
        return self._nparams
