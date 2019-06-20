from typing import List

import numpy as np
from scipy.spatial import distance_matrix

from .kernel import Kernel


class RBF(Kernel):
    def __init__(self, log_sigma: float, log_gamma: float, *, learn_sigma: bool = True) -> None:
        self.learn_sigma = learn_sigma
        num_params = 2 if self.learn_sigma else 1
        super().__init__(num_params=num_params)
        self._sigma = np.exp(log_sigma)
        self._gamma = np.exp(log_gamma)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        self._check_input_is_valid(x1, x2)
        dist = distance_matrix(x1, x2)
        kx1x2 = self._sigma * np.exp(-self._gamma * np.square(dist))
        return kx1x2

    @property
    def params(self) -> np.ndarray:
        if self.learn_sigma:
            return np.log(np.array([self._sigma, self._gamma]))
        else:
            return np.log(np.array(self._gamma))

    @params.setter
    def params(self, params: np.ndarray) -> None:
        self._check_params_are_valid(params)
        if self.learn_sigma:
            self._sigma, self._gamma = np.exp(params).copy().flatten()
        else:
            self._gamma = np.exp(params).copy().flatten()

    @property
    def true_params(self) -> np.ndarray:
        return np.exp(self.params)

    def gradients(self, x: np.ndarray) -> List[np.ndarray]:
        dist = distance_matrix(x, x)
        square_dist = np.square(dist)
        grads = []
        if self.learn_sigma:
            dsigma = self._sigma * np.exp(-self._gamma * square_dist)
            grads.append(dsigma)
        dgamma = (-self._sigma
                  * self._gamma
                  * square_dist
                  * np.exp(-self._gamma * square_dist))
        grads.append(dgamma)
        return grads

    def gradients_wrt_data(self, x: np.ndarray, n: int, dim: int) -> np.ndarray:
        """Compute the derivative matrix of the kernel wrt the data"""
        N, D = x.shape
        dist = distance_matrix(x, x)
        kxx = self._sigma * np.exp(-self._gamma * np.square(dist))
        dkdx = np.zeros((N, N))
        dkdx[n, :] = -2 * self._gamma * (x[n, dim] - x[:, dim]) * kxx[n, :]
        dkdx[:, n] = dkdx[n, :]
        return dkdx
