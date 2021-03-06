from typing import Tuple

import numpy as np
from scipy.optimize import fmin_cg

from gp.kernel import Kernel, RBF


class GP:
    _HALF_LN2PI = 0.5 * np.log(2 * np.pi)

    def __init__(self, x: np.ndarray = None, y: np.ndarray = None, kernel: Kernel = None) -> None:
        self.kernel = RBF(-1, -1) if kernel is None else kernel
        self.x, self.y = self.initialise_data(x, y)
        self.beta_exp = 50
        self.k_xx = self._compute_k_xx()
        self.chol_xx = self._compute_chol_xx()
        self.a = self._compute_a()
        self.aa_k_inv = self._compute_aa_k_inv()

    def _compute_k_xx(self) -> np.ndarray:
        k_xx = self.kernel(self.x, self.x) + np.eye(self.num_data) / self.beta_exp
        return k_xx

    def _compute_chol_xx(self) -> np.ndarray:
        try:
            chol_xx = np.linalg.cholesky(self.k_xx)
        except np.linalg.LinAlgError:
            chol_xx = np.linalg.cholesky(self.k_xx + 1e-10 * np.eye(self.num_data))
        return chol_xx

    def _compute_a(self) -> np.ndarray:
        a = np.linalg.solve(self.chol_xx.T, np.linalg.solve(self.chol_xx, self.y))
        return a

    def _compute_aa_k_inv(self) -> np.ndarray:
        k_inv = np.linalg.solve(self.chol_xx.T,
                                np.linalg.solve(self.chol_xx, np.eye(self.num_data)))
        aa_k_inv = np.matmul(self.a, self.a.T) - self.y_dim * k_inv
        return aa_k_inv

    def update(self) -> None:
        # Page 19 in GPML
        self.k_xx = self._compute_k_xx()
        self.chol_xx = self._compute_chol_xx()
        self.a = self._compute_a()

    def update_grad(self) -> None:
        self.aa_k_inv = self._compute_aa_k_inv()

    @property
    def params(self) -> np.ndarray:
        return np.hstack((self.kernel.params, np.log(self.beta_exp)))

    @params.setter
    def params(self, params: np.ndarray) -> None:
        assert params.size == self.num_params
        self.beta_exp = np.exp(params[-1])
        self.kernel.params = params[:-1]  # type: ignore
        # https://github.com/python/mypy/issues/4165

    @property
    def true_params(self) -> np.ndarray:
        return np.hstack((self.kernel.true_params, self.beta_exp))

    def posterior(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        k_xs_x = self.kernel(xs, self.x)
        k_xs_xs = self.kernel(xs, xs)
        mean = np.matmul(k_xs_x, self.a)
        v = np.linalg.solve(self.chol_xx, k_xs_x.T)
        cov = k_xs_xs - np.matmul(v.T, v) + np.eye(xs.shape[0]) / self.beta_exp
        log_likelihood = self.log_likelihood()
        return mean, cov, log_likelihood

    def log_likelihood(self, params: np.ndarray = None) -> float:
        if params is not None:
            self.params = params
        self.update()
        log_likelihood = (- 0.5 * np.trace(np.dot(self.y.T, self.a))
                          - self.y_dim * np.sum(np.log(np.diag(self.chol_xx)))
                          - self.y_dim * self.num_data * self.half_ln2pi)
        return log_likelihood

    def log_likelihood_grad(self, params: np.ndarray = None) -> np.ndarray:
        if params is not None:
            self.params = params
        self.update()
        self.update_grad()
        k_grads = [p for p in self.kernel.gradients(self.x)]
        k_grads.append(-np.eye(self.num_data) / self.beta_exp)
        grads = np.array([0.5 * np.trace(np.dot(self.aa_k_inv, k_grad)) for k_grad in k_grads])
        return grads

    def loss(self, params: np.ndarray = None) -> float:
        return -self.log_likelihood(params)

    def loss_grad(self, params: np.ndarray = None) -> np.ndarray:
        return -self.log_likelihood_grad(params)

    def optimise_hyperparameters(self) -> None:
        params, loss, *_ = fmin_cg(self.loss, x0=np.hstack(self.params), fprime=self.loss_grad,
                                   disp=False, full_output=True)
        params_restart, loss_restart, *_ = fmin_cg(self.loss, x0=-np.ones(self.num_params),
                                                   fprime=self.loss_grad, disp=False,
                                                   full_output=True)
        final_params = params if loss < loss_restart else params_restart
        _ = self.log_likelihood(final_params)

    def add_point(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.array(x).reshape(-1, self.x_dim)
        y = np.array(y).reshape(-1, self.y_dim)
        assert x.shape[0] == y.shape[0], \
            f"First dim of x {x.shape} not matching that of y {y.shape}"
        self.x = np.vstack((self.x, x))
        self.y = np.vstack((self.y, y))
        self.update()

    @staticmethod
    def initialise_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        both_are_none = x is None and y is None
        neither_are_none = x is not None and y is not None
        assert both_are_none or neither_are_none, "Provide both x and y or neither"
        if x is None:
            x = np.zeros((0, 1))
            y = np.zeros((0, 1))
        assert x.ndim == 2, f"x is {x.ndim}D; needs to be 2D array of size NxQ"
        assert y.ndim == 2, f"y is {y.ndim}D; needs to be 2D array of size NxD"
        assert x.shape[0] == y.shape[0], \
            f"First dim of x {x.shape} not matching that of y {y.shape}"
        return x, y

    @property
    def x_dim(self) -> int:
        return self.x.shape[1]

    @property
    def y_dim(self) -> int:
        return self.y.shape[1]

    @property
    def num_data(self) -> int:
        return self.y.shape[0]

    @property
    def num_params(self) -> int:
        return self.kernel.num_params + 1

    @property
    def half_ln2pi(self) -> float:
        return self._HALF_LN2PI
