import abc
from typing import List

import numpy as np


class Kernel(abc.ABC):
    """
    Base class for all kernels
    """

    def __init__(self, num_params: int) -> None:
        self._num_params = num_params

    @abc.abstractmethod
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _check_input_is_valid(x1: np.ndarray, x2: np.ndarray) -> None:
        if x1.shape[1] != x2.shape[1]:
            raise ValueError(f"Vectors must be of matching dimension, "
                             f"but x1.shape = {x1.shape} and x2.shape = {x2.shape}")

    def _check_params_are_valid(self, params: np.ndarray) -> None:
        if params.size != self.num_params:
            raise ValueError(f"Provided {params.size} params; must be {self.num_params}")

    @property  # type: ignore  # https://github.com/python/mypy/issues/4165
    @abc.abstractmethod
    def params(self) -> np.ndarray:
        raise NotImplementedError

    @params.setter  # type: ignore  # https://github.com/python/mypy/issues/4165
    @abc.abstractmethod
    def params(self, params: np.ndarray) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def true_params(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def gradients(self, x: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def gradients_wrt_data(self, x: np.ndarray, n: int, dim: int) -> np.ndarray:
        raise NotImplementedError

    @property
    def num_params(self) -> int:
        return self._num_params
