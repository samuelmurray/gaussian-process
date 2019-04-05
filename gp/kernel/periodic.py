from typing import List

import numpy as np
from scipy.spatial import distance_matrix

from .kernel import Kernel


class Periodic(Kernel):
    def __init__(self, sigma: float, gamma: float, period: float,
                 learn_sigma: bool = True) -> None:
        self.learn_sigma = learn_sigma
        super().__init__(nparams=3)
        self._sigma_exp = np.exp(sigma)
        self._gamma_exp = np.exp(gamma)
        self._period_exp = np.exp(period)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        self._check_input_is_valid(x1, x2)
        dist = distance_matrix(x1, x2)
        abs_dist = np.abs(dist)
        R = np.square(np.sin(abs_dist / self._period_exp))
        kx1x2 = self._sigma_exp * np.exp(-self._gamma_exp * R)
        return kx1x2

    def set_params(self, params: np.ndarray) -> None:
        super().set_params(params)
        if self.learn_sigma:
            self._sigma_exp, self._gamma_exp, self._period_exp = np.exp(params).copy().flatten()
        else:
            self._gamma_exp, self._period_exp = np.exp(params).copy().flatten()

    def get_params(self) -> np.ndarray:
        if self.learn_sigma:
            return np.log(np.array([self._sigma_exp, self._gamma_exp, self._period_exp]))
        else:
            return np.log(np.array([self._gamma_exp, self._period_exp]))

    def get_true_params(self) -> np.ndarray:
        return np.exp(self.get_params())

    def gradients(self, x: np.ndarray) -> List[np.ndarray]:
        dist = distance_matrix(x, x)
        abs_dist = np.abs(dist)
        R = np.square(np.sin(abs_dist / self._period_exp))
        grads = []
        if self.learn_sigma:
            dsigma = self._sigma_exp * np.exp(-self._gamma_exp * R)
            grads.append(dsigma)
        dgamma = -self._sigma_exp * self._gamma_exp * R * np.exp(-self._gamma_exp * R)
        grads.append(dgamma)
        dperiod = (2
                   * self._sigma_exp
                   * self._gamma_exp
                   * abs_dist
                   * np.sin(abs_dist / self._period_exp)
                   * np.cos(abs_dist / self._period_exp)
                   * np.exp(-self._gamma_exp * R)
                   / self._period_exp)
        grads.append(dperiod)
        return grads

    def gradients_wrt_data(self, x: np.ndarray, n: int, dim: int) -> np.ndarray:
        raise NotImplementedError
