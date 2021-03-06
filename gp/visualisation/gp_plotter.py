from typing import Callable

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib import pyplot as plt

from gp.model.gp import GP


class GPPlotter:
    def __init__(self, gp: GP, func: Callable[[np.ndarray], np.ndarray] = None) -> None:
        self.func = func
        self.gp = gp
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.xs = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)

    def plot_prior_sample(self, nsamples: int) -> None:
        n = self.xs.shape[0]
        mean = np.zeros(n)
        cov = self.gp.kernel(self.xs, self.xs)
        f = np.random.multivariate_normal(mean, cov, size=nsamples).T
        plt.plot(self.xs, f)
        plt.show()

    def plot_posterior(self) -> None:
        mean, cov, log_likelihood = self.gp.posterior(self.xs)
        var = np.diag(cov).reshape(-1, 1)
        std = np.sqrt(var)  # FIXME: In some cases var will be small negative, gives RuntimeWarning
        params = self.gp.true_params
        plt.cla()
        plt.plot(self.xs, mean)
        plt.plot(self.xs, mean + std, 'k')
        plt.plot(self.xs, mean - std, 'k')
        plt.scatter(self.gp.x, self.gp.y)
        self.ax.set_title(f"Log likelihood: {log_likelihood}\n Parameter values: {params}")
        self.ax.set_xlim([-np.pi, np.pi])
        self.ax.set_ylim([-1.1, 1.1])
        self.fig.canvas.draw()

    def plot_posterior_sample(self, nsamples: int) -> None:
        mean, cov, log_likelihood = self.gp.posterior(self.xs)
        f = np.random.multivariate_normal(mean[:, 0], cov, size=nsamples).T
        plt.cla()
        plt.plot(self.xs, f)
        plt.scatter(self.gp.x, self.gp.y)
        self.ax.set_title(f"Log likelihood: {log_likelihood}")
        self.fig.canvas.draw()

    def onclick(self, event: MouseEvent) -> None:
        x = event.xdata
        y = event.ydata if self.func is None else self.func(x)
        self.gp.add_point(x, y)
        self.gp.optimise_hyperparameters()
        self.plot_posterior()

    def add_point(self, x: np.ndarray, y: np.ndarray) -> None:
        self.gp.add_point(x, y)
