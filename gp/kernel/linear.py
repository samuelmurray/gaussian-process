import numpy as np

from .kernel import Kernel


class Linear(Kernel):
    def __init__(self, sigma, learn_sigma=True) -> None:
        self.learn_sigma = learn_sigma
        nparams = 1 if self.learn_sigma else 0
        super().__init__(nparams=nparams)
        self._sigma_exp = np.exp(sigma)

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        super().__call__(x1, x2)
        prod = np.dot(x1, x2.T)
        kx1x2 = self._sigma_exp * prod
        return kx1x2

    def set_params(self, params) -> None:
        super().set_params(params)
        if self.learn_sigma:
            self._sigma_exp = np.exp(params).flatten()

    def get_params(self):
        params = np.log(np.array(self._sigma_exp)) if self.learn_sigma else []
        return params

    def get_true_params(self):
        return np.exp(self.get_params())

    def gradients(self, x: np.ndarray):
        grads = []
        if self.learn_sigma:
            prod = np.dot(x, x.T)
            dsigma = self._sigma_exp * prod
            grads.append(dsigma)
        return grads

    def gradients_wrt_data(self, x: np.ndarray, n=None, dim=None):
        raise NotImplementedError
