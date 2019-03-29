import numpy as np
from scipy.spatial import distance_matrix

from . import Kernel


class Periodic(Kernel):
    def __init__(self, sigma, gamma, period):
        super().__init__(nparams=3)
        self._sigma_exp = np.exp(sigma)
        self._gamma_exp = np.exp(gamma)
        self._period_exp = np.exp(period)

    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        dist = distance_matrix(x1, x2)
        abs_dist = np.abs(dist)
        R = np.square(np.sin(abs_dist / self._period_exp))
        kx1x2 = self._sigma_exp * np.exp(-self._gamma_exp * R)
        return kx1x2

    def set_params(self, params):
        super().set_params(params)
        self._sigma_exp, self._gamma_exp, self._period_exp = np.exp(params).copy().flatten()

    def get_params(self):
        return np.log(np.array([self._sigma_exp, self._gamma_exp, self._period_exp]))

    def get_true_params(self):
        return np.exp(self.get_params())

    def gradients(self, x):
        dist = distance_matrix(x, x)
        abs_dist = np.abs(dist)
        R = np.square(np.sin(abs_dist / self._period_exp))
        dsigma = self._sigma_exp * np.exp(-self._gamma_exp * R)
        dgamma = -self._sigma_exp * self._gamma_exp * R * np.exp(-self._gamma_exp * R)
        dperiod = (2
                   * self._sigma_exp
                   * self._gamma_exp
                   * abs_dist
                   * np.sin(abs_dist / self._period_exp)
                   * np.cos(abs_dist / self._period_exp)
                   * np.exp(-self._gamma_exp * R)
                   / self._period_exp)
        return dsigma, dgamma, dperiod

    def gradients_wrt_data(self, x, n=None, dim=None):
        raise NotImplementedError
