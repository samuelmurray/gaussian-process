import numpy as np
from scipy.spatial import distance_matrix

from .kernel import Kernel


class RBF(Kernel):
    def __init__(self, sigma, gamma, learn_sigma=True):
        self.learn_sigma = learn_sigma
        nparams = 2 if self.learn_sigma else 1
        super().__init__(nparams=nparams)
        self._sigma_exp = np.exp(sigma)
        self._gamma_exp = np.exp(gamma)

    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        dist = distance_matrix(x1, x2)
        kx1x2 = self._sigma_exp * np.exp(-self._gamma_exp * np.square(dist))
        return kx1x2

    def set_params(self, params):
        super().set_params(params)
        if self.learn_sigma:
            self._sigma_exp, self._gamma_exp = np.exp(params).copy().flatten()
        else:
            self._gamma_exp = np.exp(params).copy().flatten()

    def get_params(self):
        if self.learn_sigma:
            return np.log(np.array([self._sigma_exp, self._gamma_exp]))
        else:
            return np.log(np.array(self._gamma_exp))

    def get_true_params(self):
        return np.exp(self.get_params())

    def gradients(self, x):
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

    def gradients_wrt_data(self, x, n=None, dim=None):
        """Compute the derivative matrix of the kernel wrt the data.
        This returns a list of matrices: each matrix is NxN, and there are N*D of them!"""
        N, D = x.shape
        dist = distance_matrix(x, x)
        kxx = self._sigma_exp * np.exp(-self._gamma_exp * np.square(dist))

        if (n is None) and (dim is None):  # calculate all gradients
            dkdx_list = []
            for n_iter in range(N):
                for d_iter in range(D):
                    dkdx = np.zeros((N, N))
                    dkdx[n_iter, :] = (-2
                                       * self._gamma_exp
                                       * (x[n_iter, d_iter] - x[:, d_iter])
                                       * kxx[n_iter, :])
                    dkdx[:, n_iter] = dkdx[n_iter, :]
                    dkdx_list.append(dkdx.copy())
            return dkdx_list
        else:
            dkdx = np.zeros((N, N))
            dkdx[n, :] = -2 * self._gamma_exp * (x[n, dim] - x[:, dim]) * kxx[n, :]
            dkdx[:, n] = dkdx[n, :]
            return dkdx
