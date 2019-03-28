from matplotlib import pyplot as plt

from gp.visualisation import util, GP_Plotter
from gp.kernel import *
from gp.model import GP

if __name__ == "__main__":
    func = None  # This can be changed to e.g. util.sin
    kern = RBF(-1, -1)
    gp = GP(kern=kern)
    gplot = GP_Plotter(gp, func)
    gplot.plot_posterior()
    plt.show()
