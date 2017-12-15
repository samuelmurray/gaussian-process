import numpy as np
from matplotlib import pyplot as plt

from model import GPLVM
from kernel import *

if __name__ == "__main__":
    kern = RBF(0, 0)
    n = 50
    t = np.linspace(0, 2 * np.pi, n)
    x = np.array([np.cos(t), np.sin(t)]).T
    mean = np.zeros(n)
    cov = kern(x, x)
    f = np.random.multivariate_normal(mean, cov, size=10).T
    m = GPLVM(f, initialise_by_pca=False)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(f)
    ax[0, 1].plot(m.x[:, 0], m.x[:, 1])
    #m.optimise_latents()
    m.optimise(5, learn_hyperparameters=True)
    ax[1, 0].scatter(m.x[:, 0], m.x[:, 1])
    ax[1, 0].plot(m.x[:, 0], m.x[:, 1])
    plt.show()
