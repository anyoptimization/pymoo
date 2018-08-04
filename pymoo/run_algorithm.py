import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from pymop.problems.dtlz import DTLZ2
from pymop.problems.rastrigin import Rastrigin
from pymop.problems.zdt import ZDT1

if __name__ == '__main__':

    def fun(x):
        return ZDT1(n_var=30).evaluate(x)

        n_samples = x.shape[0]

        f = np.full((n_samples, 2), np.inf)
        f[:, 0] = np.sum(np.square(x - 0.25), axis=1)
        f[:, 1] = np.sum(np.square(x - 0.75), axis=1)

        g = np.full((n_samples, 1), np.inf)
        g[:, 0] = 0.1 - f[:, 0]

        return f, g


    from pymoo.optimize import minimize

    res = minimize(Rastrigin(n_var=30), method="de", termination=('n_eval', 90000), disp=True)

    plt.scatter(res['F'][:,0], res['F'][:,1])
    plt.show()

    exit()

    # load the problem instance
    # from pymop.problems.zdt import ZDT4
    # problem = ZDT4()
    # problem = Rastrigin(n_var=30)
    problem = DTLZ2(n_var=10)

    # create the algorithm instance by specifying the intended parameters
    from pymoo.algorithms.nsga3 import NSGA3

    algorithm = NSGA3("real", pop_size=100, verbose=True)

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
    print(F)

    scatter_plot = True
    save_animation = False

    # get the problem dimensionality
    is_2d = problem.n_obj == 2
    is_3d = problem.n_obj == 3

    if scatter_plot and is_2d:
        plt.scatter(F[:, 0], F[:, 1])
        plt.show()

    if scatter_plot and is_3d:
        from mpl_toolkits.mplot3d import Axes3D

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
        plt.scatter(pf[:, 0], pf[:, 1], label='Pareto Front', s=60, facecolors='none', edgecolors='r')
        scat = plt.scatter(_F[:, 0], _F[:, 1])


        def update(frame_number):
            _F = history[frame_number]['F']
            scat.set_offsets(_F)

            # get the bounds for plotting and add padding
            min = np.min(_F, axis=0) - 0.1
            max = np.max(_F, axis=0) + 0.

            # set the scatter object with padding
            ax.set_xlim(min[0], max[0])
            ax.set_ylim(min[1], max[1])


        # create the animation
        ani = animation.FuncAnimation(fig, update, frames=range(n_gen))

        # write the file
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=6, bitrate=1800)
        ani.save('%s.mp4' % problem.name(), writer=writer)
