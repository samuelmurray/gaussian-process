from typing import List

import numpy as np
from scipy.spatial import distance_matrix

from .kernel import Kernel


class Periodic(Kernel):
    def __init__(self, log_sigma: float, log_gamma: float, log_period: float, *,
                 learn_sigma: bool = True) -> None:
        self.learn_sigma = learn_sigma
        num_params = 3 if self.learn_sigma else 2
        super().__init__(num_params=num_params)
        self._sigma = np.exp(log_sigma)
        self._gamma = np.exp(log_gamma)
        self._period = np.exp(log_period)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        self._check_input_is_valid(x1, x2)
        dist = distance_matrix(x1, x2)
        abs_dist = np.abs(dist)
        R = np.square(np.sin(abs_dist / self._period))
        kx1x2 = self._sigma * np.exp(-self._gamma * R)
        return kx1x2

    @property
    def params(self) -> np.ndarray:
        if self.learn_sigma:
            return np.log(np.array([self._sigma, self._gamma, self._period]))
        else:
            return np.log(np.array([self._gamma, self._period]))

    @params.setter
    def params(self, params: np.ndarray) -> None:
        self._check_params_are_valid(params)
        if self.learn_sigma:
            self._sigma, self._gamma, self._period = np.exp(params).copy().flatten()
        else:
            self._gamma, self._period = np.exp(params).copy().flatten()

    @property
    def true_params(self) -> np.ndarray:
        return np.exp(self.params)

    def gradients(self, x: np.ndarray) -> List[np.ndarray]:
        dist = distance_matrix(x, x)
        abs_dist = np.abs(dist)
        R = np.square(np.sin(abs_dist / self._period))
        grads = []
        if self.learn_sigma:
            dsigma = self._sigma * np.exp(-self._gamma * R)
            grads.append(dsigma)
        dgamma = -self._sigma * self._gamma * R * np.exp(-self._gamma * R)
        grads.append(dgamma)
        dperiod = (2
                   * self._sigma
                   * self._gamma
                   * abs_dist
                   * np.sin(abs_dist / self._period)
                   * np.cos(abs_dist / self._period)
                   * np.exp(-self._gamma * R)
                   / self._period)
        grads.append(dperiod)
        return grads

    def gradients_wrt_data(self, x: np.ndarray, n: int, dim: int) -> np.ndarray:
        raise NotImplementedError
