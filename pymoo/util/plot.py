import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pymop.ackley import Ackley


def plot_problem_surface(problem):

    # number of points to be sampled
    n_samples = 300

    if problem.n_var == 1 and problem.n_obj == 1:
        X = np.linspace(problem.xl[0], problem.xu[0], num=n_samples)[:,None]
        Y = problem.evaluate(X, return_constraints=0)
        plt.plot(X,Y)

    elif problem.n_var == 2 and problem.n_obj == 1:

        fig = plt.figure(1)
        plot = fig.add_subplot(111, projection='3d')

        X_range = np.linspace(problem.xl[0], problem.xu[0], num=n_samples)
        Y_range = np.linspace(problem.xl[1], problem.xu[1], num=n_samples)
        X, Y = np.meshgrid(X_range, Y_range)

        A = np.zeros((n_samples * n_samples, 2))
        counter = 0
        for i, x in enumerate(X_range):
            for j, y in enumerate(Y_range):
                A[counter, 0] = x
                A[counter, 1] = y
                counter += 1


        F = np.reshape(problem.evaluate(A, return_constraints=0), (n_samples, n_samples))

        # Plot the surface.
        # CS = plot.contour(X, Y, F)
        # plt.colorbar(CS)

        # plot.plot_countour(X, Y, F, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plot.plot_wireframe(X, Y, F)

    else:
        raise Exception("Can only plot problems with less than 2 variables and ")

    plt.show()


if __name__ == '__main__':
    plot_problem_surface(Ackley(n_var=1))



