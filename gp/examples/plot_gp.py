from matplotlib import pyplot as plt

from gp.visualisation import GPPlotter
from gp.kernel import RBF
from gp.model import GP

if __name__ == "__main__":
    func = None  # This can be changed to e.g. util.sin
    kern = RBF(-1, -1)
    gp = GP(kern=kern)
    gplot = GPPlotter(gp, func)
    gplot.plot_posterior()
    plt.show()
