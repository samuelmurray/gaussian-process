import numpy as np
from scipy.optimize import fmin_cg
from sklearn.decomposition import PCA

from ..kernel import RBF
from . import GP


class GPLVM(GP):
    def __init__(self, y, kern=None, initialise_by_pca=False):
        if kern is None:
            kern = RBF(-1, -1)
        latent_dim = 2
        if initialise_by_pca:
            pca = PCA(latent_dim)
            x = pca.fit_transform(y)
        else:
            x = np.random.normal(0, 1, size=(y.shape[0], latent_dim))
        super().__init__(x, y, kern=kern)

    def log_joint(self, xx, n):
        """The log joint: log p(y,x)=log p(y|x) + log p(x). We find MAP solution wrt x.
        log p(y|x) is the likelihood from GP regression; p(x) is Gaussian prior on x.
        Used when changing the ith latent variable to xx"""
        self.x[n] = xx
        self.update()
        log_likelihood = self.log_likelihood()
        log_prior = - 0.5 * np.sum(np.square(self.x)) - self.xdim * self.n * self.half_ln2pi
        return log_likelihood + log_prior

    def log_joint_grad(self, xx, n):
        """The gradient of the log joint: d/dx {p(y,x)}. Used to find MAP solution of joint."""
        self.x[n] = xx
        self.update()
        self.update_grad()
        k_grads = [self.kern.gradients_wrt_data(self.x, n, dim) for dim in range(self.xdim)]
        grads = np.array([0.5 * np.trace(np.dot(self.aa_k_inv, k_grad)) for k_grad in k_grads])
        return grads - xx

    def joint_loss(self, xx, n):
        return -self.log_joint(xx, n)

    def joint_loss_grad(self, xx, n):
        return -self.log_joint_grad(xx, n)

    def optimise(self, n_iter, learn_hyperparameters=True):
        from matplotlib import pyplot as plt
        for i in range(n_iter):
            print(f"Iteration {i}")
            self.optimise_latents()
            if learn_hyperparameters:
                self.optimise_hyperparameters()
            plt.plot(self.x[:, 0], self.x[:, 1])
            plt.title("Optimisation of X")
            plt.legend(range(n_iter))

    def optimise_latents(self, n_iter=1):
        """Direct optimisation of the latents variables."""
        for iteration in range(n_iter):
            xtemp = np.zeros(self.x.shape)
            for i, yy in enumerate(self.y):
                original_x = self.x[i].copy()
                xopt, loss, *_ = fmin_cg(self.joint_loss, self.x[i], fprime=self.joint_loss_grad, args=(i,), disp=False,
                                         full_output=True)
                self.x[i] = original_x
                xtemp[i] = xopt
            self.x = xtemp.copy()

    @property
    def latent_dim(self):
        return self.xdim
