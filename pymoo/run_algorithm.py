import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from pymop.zdt import ZDT

from pymoo.algorithms.NSGAIII import NSGAIII
from pymoo.algorithms.RNSGAIII import RNSGAIII

if __name__ == '__main__':

    # load the problem instance
    class ZDT1(ZDT):
        def __init__(self, n_var=30):
            ZDT.__init__(self, n_var)

        def calc_pareto_front(self):
            x1 = np.arange(0, 1.01, 0.01)
            return np.array([x1, 1 - np.sqrt(x1)]).T

        def _evaluate(self, x, f):
            f[:, 0] = x[:, 0]
            f[:, 0] = -100 + (f[:, 0] * 50)
            g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
            f[:, 1] = g * (1 - np.power((x[:, 0] / g), 0.5))

        def pareto_front(self):
            pf = super().pareto_front()
            pf[:, 0] = -100 + (pf[:, 0] * 50)
            return pf


    problem = ZDT1()

    ref_points = np.array([[-100, 0.6], [-60, 0.05]])
    plt.scatter(ref_points[:,0], ref_points[:,1])

    algorithm = RNSGAIII("real", ref_points, verbose=True, epsilon=0.001, weights=None)
    algorithm = NSGAIII("real", pop_size=200)

    start_time = time.time()

    # save the history in an object to observe the convergence over generations
    history = []

    # number of generations to run it
    n_gen = 200

    # solve the problem and return the results
    X, F, G = algorithm.solve(problem,
                              evaluator=(100 * n_gen),
                              seed=2,
                              return_only_feasible=False,
                              return_only_non_dominated=False,
                              history=history)

    print("--- %s seconds ---" % (time.time() - start_time))

    print(X)
    print(F)

    scatter_plot = True
    save_animation = False

    # get the problem dimensionality
    is_2d = problem.n_obj == 2
    is_3d = problem.n_obj == 3

    if scatter_plot and is_2d:
        pf = problem.pareto_front()
        plt.scatter(pf[:, 0], pf[:, 1], label='Pareto Front', s=60, facecolors='none', edgecolors='r')
        plt.scatter(F[:, 0], F[:, 1], color='b')
        plt.show()

    if scatter_plot and is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(F[:, 0], F[:, 1], F[:, 2])
        plt.show()

    # create an animation to watch the convergence over time
    if is_2d and save_animation:

        fig = plt.figure()
        ax = plt.gca()

        _F = history[0]['F']
        pf = problem.pareto_front()
        plt.scatter(pf[:,0], pf[:,1], label='Pareto Front', s=60, facecolors='none', edgecolors='r')
        scat = plt.scatter(_F[:, 0], _F[:, 1])


        def update(frame_number):
            _F = history[frame_number]['F']
            scat.set_offsets(_F)

            # get the bounds for plotting and add padding
            min = np.min(_F, axis=0) - 0.1
            max = np.max(_F, axis=0) + 0.1

            # set the scatter object with padding
            ax.set_xlim(min[0], max[0])
            ax.set_ylim(min[1], max[1])


        # create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(history))

        # write the file
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=6, bitrate=1800)
        ani.save('%s.mp4' % problem.name(), writer=writer)
