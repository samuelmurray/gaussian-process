import numpy as np
from matplotlib import pyplot as plt

import util
from kernel import *
from model import GP


class GPplotter:
    def __init__(self, gp, func=None):
        self.func = func
        self.gp = gp
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.xs = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)

    def plot_prior_sample(self, nsamples):
        n = self.xs.shape[0]
        mean = np.zeros(n)
        cov = self.gp.kern(self.xs, self.xs)
        f = np.random.multivariate_normal(mean, cov, size=nsamples).T
        plt.plot(self.xs, f)
        plt.show()

    def plot_posterior(self):
        mean, cov, log_likelihood = self.gp.posterior(self.xs)
        var = np.diag(cov).reshape(-1, 1)
        std = np.sqrt(var)  # FIXME: In some cases var will be small negative, gives RuntimeWarning.
        try:
            params = self.gp.get_true_params()
        except NotImplementedError:
            params = self.gp.get_params()  # TODO: Maybe better to force implementation of get_true_params?
        plt.cla()
        self.ax.plot(self.xs, mean)
        self.ax.plot(self.xs, mean + std, 'k')
        self.ax.plot(self.xs, mean - std, 'k')
        self.ax.scatter(self.gp.x, self.gp.y)
        self.ax.set_title(f"Log likelihood: {log_likelihood}\n Parameter values: {params}")
        self.fig.canvas.draw()

    def plot_posterior_sample(self, nsamples):
        mean, cov, log_likelihood = self.gp.posterior(self.xs)
        f = np.random.multivariate_normal(mean[:, 0], cov, size=nsamples).T
        plt.cla()
        self.ax.plot(self.xs, f)
        self.ax.scatter(self.gp.x, self.gp.y)
        self.ax.set_title(f"Log likelihood: {log_likelihood}")
        self.fig.canvas.draw()

    def onclick(self, event):
        x = event.xdata
        y = event.ydata if self.func is None else self.func(x)
        #self.gp.add_point(x, y)
        self.gp.optimise_hyperparameters()
        # self.plot_posterior_sample(10)
        self.plot_posterior()


def dkl_step():
    y = np.array(
        [10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 11, 12, 12,
         12, 12, 13, 13, 13, 13, 13, 12, 13, 13, 13, 12, 13, 9, 9, 9, 8, 8, 9, 9, 9, 8, 8, 9, 9, 8, 8, 8, 9, 8, 7, 7, 7,
         8, 8, 7, 7, 7, 7, 8, 7, 7]).reshape(-1, 1)
    x = np.linspace(-1, 1, len(y)).reshape(-1, 1)
    return x, y


if __name__ == "__main__":
    func = util.sin
    x = np.linspace(-1.5, 1.5, 7).reshape(-1, 1)
    y = func(x)
    #kern = RBF(0, -1, True)
    kern = Linear(-1)
    gp = GP(x, y, kern=kern)
    #gp.optimise_hyperparameters()
    gplot = GPplotter(gp, func=func)
    gplot.plot_posterior()
    # gplot.plot_posterior_sample(5)
    plt.show()
