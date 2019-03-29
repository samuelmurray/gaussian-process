import numpy as np
from scipy.spatial import distance_matrix

from .kernel import Kernel


class Exponential(Kernel):
    def __init__(self, sigma, gamma):
        super().__init__(nparams=2)
        self._sigma_exp = np.exp(sigma)
        self._gamma_exp = np.exp(gamma)

    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        dist = distance_matrix(x1, x2)
        kx1x2 = self._sigma_exp * np.exp(-self._gamma_exp * np.abs(dist))
        return kx1x2

    def set_params(self, params):
        super().set_params(params)
        self._sigma_exp, self._gamma_exp = np.exp(params).copy().flatten()

    def get_params(self):
        return np.log(np.array([self._sigma_exp, self._gamma_exp]))

    def get_true_params(self):
        return np.exp(self.get_params())

    def gradients(self, x):
        dist = distance_matrix(x, x)
        abs_dist = np.abs(dist)
        dsigma = self._sigma_exp * np.exp(-self._gamma_exp * abs_dist)
        dgamma = -self._sigma_exp * self._gamma_exp * abs_dist * np.exp(-self._gamma_exp * abs_dist)
        return dsigma, dgamma

    def gradients_wrt_data(self, x, n=None, dim=None):
        raise NotImplementedError
