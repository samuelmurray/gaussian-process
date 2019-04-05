from typing import List

import numpy as np
from scipy.spatial import distance_matrix

from .kernel import Kernel


class RBF(Kernel):
    def __init__(self, sigma: float, gamma: float, learn_sigma: bool = True) -> None:
        self.learn_sigma = learn_sigma
        nparams = 2 if self.learn_sigma else 1
        super().__init__(nparams=nparams)
        self._sigma_exp = np.exp(sigma)
        self._gamma_exp = np.exp(gamma)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        self._check_input_is_valid(x1, x2)
        dist = distance_matrix(x1, x2)
        kx1x2 = self._sigma_exp * np.exp(-self._gamma_exp * np.square(dist))
        return kx1x2

    def set_params(self, params: np.ndarray) -> None:
        super().set_params(params)
        if self.learn_sigma:
            self._sigma_exp, self._gamma_exp = np.exp(params).copy().flatten()
        else:
            self._gamma_exp = np.exp(params).copy().flatten()

    def get_params(self) -> np.ndarray:
        if self.learn_sigma:
            return np.log(np.array([self._sigma_exp, self._gamma_exp]))
        else:
            return np.log(np.array(self._gamma_exp))

    def get_true_params(self) -> np.ndarray:
        return np.exp(self.get_params())

    def gradients(self, x: np.ndarray) -> List[np.ndarray]:
        dist = distance_matrix(x, x)
        square_dist = np.square(dist)
        grads = []
        if self.learn_sigma:
            dsigma = self._sigma_exp * np.exp(-self._gamma_exp * square_dist)
            grads.append(dsigma)
        dgamma = (-self._sigma_exp
                  * self._gamma_exp
                  * square_dist
                  * np.exp(-self._gamma_exp * square_dist))
        grads.append(dgamma)
        return grads

    def gradients_wrt_data(self, x: np.ndarray, n: int, dim: int) -> np.ndarray:
        """Compute the derivative matrix of the kernel wrt the data"""
        N, D = x.shape
        dist = distance_matrix(x, x)
        kxx = self._sigma_exp * np.exp(-self._gamma_exp * np.square(dist))
        dkdx = np.zeros((N, N))
        dkdx[n, :] = -2 * self._gamma_exp * (x[n, dim] - x[:, dim]) * kxx[n, :]
        dkdx[:, n] = dkdx[n, :]
        return dkdx
