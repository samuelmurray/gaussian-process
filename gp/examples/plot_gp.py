import numpy as np
from matplotlib import pyplot as plt

from gp.visualisation import util, GP_Plotter
from gp.kernel import *
from gp.model import GP


def dkl_step():
    y = np.array(
        [10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 11, 12, 12,
         12, 12, 13, 13, 13, 13, 13, 12, 13, 13, 13, 12, 13, 9, 9, 9, 8, 8, 9, 9, 9, 8, 8, 9, 9, 8, 8, 8, 9, 8, 7, 7, 7,
         8, 8, 7, 7, 7, 7, 8, 7, 7]).reshape(-1, 1)
    x = np.linspace(-1, 1, len(y)).reshape(-1, 1)
    return x, y


if __name__ == "__main__":
    func = util.sin
    kern = RBF(-1, -1)
    gp = GP(kern=kern)
    gplot = GP_Plotter(gp, func=func)
    gplot.plot_prior_sample(5)
    plt.show()
