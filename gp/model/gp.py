import numpy as np
from scipy.optimize import fmin_cg

from ..kernel import RBF


class GP:
    _half_ln2pi = 0.5 * np.log(2 * np.pi)

    def __init__(self, x=None, y=None, kern=None):
        self.kern = RBF(-1, -1) if kern is None else kern
        self.x, self.y = self.initialise_data(x, y)
        self.beta_exp = 50
        self.K = None
        self.L = None
        self.a = None
        self.aa_k_inv = None
        self.update()

    def update(self):
        # Page 19 in GPML
        self.K = self.kern(self.x, self.x) + np.eye(self.n) / self.beta_exp
        try:
            self.L = np.linalg.cholesky(self.K)
        except np.linalg.LinAlgError:
            # print(f"K is not a PSD matrix! :(")  # TODO: How to handle this?
            # print(f"Maybe because of beta={self.beta_exp}?")
            self.L = np.linalg.cholesky(self.K + 1e-10 * np.eye(self.n))
        self.a = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def update_grad(self):
        k_inv = np.linalg.solve(self.L.T, np.linalg.solve(self.L, np.eye(self.n)))
        self.aa_k_inv = np.matmul(self.a, self.a.T) - self.ydim * k_inv

    def set_params(self, params):
        assert params.size == self.nparams
        self.beta_exp = np.exp(params[-1])
        self.kern.set_params(params[:-1])

    def get_params(self):
        return np.hstack((self.kern.get_params(), np.log(self.beta_exp)))

    def get_true_params(self):
        return np.hstack((self.kern.get_true_params(), self.beta_exp))

    def posterior(self, xs):
        k_xs_x = self.kern(xs, self.x)
        k_xs_xs = self.kern(xs, xs)
        mean = np.matmul(k_xs_x, self.a)
        v = np.linalg.solve(self.L, k_xs_x.T)
        cov = k_xs_xs - np.matmul(v.T, v) + np.eye(xs.shape[0]) / self.beta_exp
        log_likelihood = self.log_likelihood()
        return mean, cov, log_likelihood

    def log_likelihood(self, params=None):
        if params is not None:
            self.set_params(params)
        self.update()
        log_likelihood = - 0.5 * np.trace(np.dot(self.y.T, self.a)) - self.ydim * np.sum(np.log(np.diag(self.L))) \
                         - self.ydim * self.n * self.half_ln2pi
        return log_likelihood

    def log_likelihood_grad(self, params=None):
        if params is not None:
            self.set_params(params)
        self.update()
        self.update_grad()
        k_grads = [p for p in self.kern.gradients(self.x)]
        k_grads.append(-np.eye(self.n) / self.beta_exp)
        grads = np.array([0.5 * np.trace(np.dot(self.aa_k_inv, k_grad)) for k_grad in k_grads])
        return grads

    def loss(self, params=None):
        return -self.log_likelihood(params)

    def loss_grad(self, params=None):
        return -self.log_likelihood_grad(params)

    def optimise_hyperparameters(self):
        params, loss, *_ = fmin_cg(self.loss, x0=np.hstack((self.get_params())), fprime=self.loss_grad, disp=False,
                                   full_output=True)
        params_restart, loss_restart, *_ = fmin_cg(self.loss, x0=-np.ones(self.nparams), fprime=self.loss_grad,
                                                   disp=False, full_output=True)
        final_params = params if loss < loss_restart else params_restart
        final_ll = self.log_likelihood(final_params)

    def add_point(self, x, y):
        x = np.array(x).reshape(-1, self.xdim)
        y = np.array(y).reshape(-1, self.ydim)
        assert x.shape[0] == y.shape[0], f"First dim of x {x.shape} not matching that of y {y.shape}"
        self.x = np.vstack((self.x, x))
        self.y = np.vstack((self.y, y))
        self.update()

    @staticmethod
    def initialise_data(x, y):
        assert (x is None and y is None) or (x is not None and y is not None), "Provide both x and y or neither"
        if x is None:
            x = np.zeros((0, 1))
            y = np.zeros((0, 1))
        assert x.ndim == 2, f"x is {x.ndim}D; needs to be 2D array of size NxQ"
        assert y.ndim == 2, f"y is {y.ndim}D; needs to be 2D array of size NxD"
        assert x.shape[0] == y.shape[0], f"First dim of x {x.shape} not matching that of y {y.shape}"
        return x, y

    @property
    def xdim(self):
        return self.x.shape[1]

    @property
    def ydim(self):
        return self.y.shape[1]

    @property
    def n(self):
        return self.y.shape[0]

    @property
    def nparams(self):
        return self.kern.nparams + 1

    @property
    def half_ln2pi(self):
        return self._half_ln2pi
