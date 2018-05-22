import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.distance import squareform

from pyop.problems.griewank import Griewank
from pyop.problems.zakharov import Zakharov


def plot_func_3d(plot, func_eval, xlim=(0, 1), ylim=(0, 1)):

    n = 100

    X_range = np.linspace(xlim[0], xlim[1], num=n)
    Y_range = np.linspace(ylim[0], ylim[1], num=n)
    X, Y = np.meshgrid(X_range, Y_range)

    A = np.zeros((n*n,2))
    counter = 0
    for i, x in enumerate(X_range):
        for j, y in enumerate(Y_range):
            A[counter, 0] = x
            A[counter, 1] = y
            counter += 1
    F = np.reshape(func_eval(A), (n,n))

    # Plot the surface.
    # CS = plot.contour(X, Y, F)
    # plt.colorbar(CS)

    # plot.plot_wireframe(X, Y, F)
    plot.plot_surface(X, Y, F, cmap=cm.coolwarm, linewidth=0, antialiased=False)


if __name__ == '__main__':

    problem = Zakharov(n_var=2)

    fig = plt.figure(1)
    plot = fig.add_subplot(111, projection='3d')
    plot_func_3d(plot, lambda x: problem.evaluate(x)[0], xlim=[np.min(problem.xl), np.min(problem.xu)], ylim=[np.min(problem.xl), np.min(problem.xu)])
    plt.show()

    X = np.linspace(np.min(problem.xl), np.min(problem.xu), num=1000)
    Y, _ = Griewank(n_var=1).evaluate(X[:, None])
    plt.plot(X, Y)
    plt.show()
    print()
