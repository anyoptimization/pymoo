import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from pymoo.algorithms.RNSGAII import RNSGAII
from pymoo.model.evaluator import Evaluator

from pymop.zdt import ZDT1

if __name__ == '__main__':

    problem = ZDT1(n_var=10)
    start_time = time.time()

    # run the algorithm
    # algorithm = SingleObjectiveGeneticAlgorithm("binary")
    ref_points = np.array([[0.0, 1.0], [0.5, 0.5]])
    algorithm = RNSGAII("real", ref_points=ref_points, epsilon=0.01, pop_size=50, verbose=True)
    eval = Evaluator(50 * 100)
    X, F, G = algorithm.solve(problem, evaluator=eval, seed=1, return_only_feasible=False, return_only_non_dominated=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(F)


    plot = True
    movie = True
    is_2d = F.shape[1] == 2
    is_3d = F.shape[1] == 3

    plt.scatter(ref_points[:,0],ref_points[:,1])

    if plot and is_2d:
        plt.scatter(F[:, 0], F[:, 1])
        plt.show()

    if plot and is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(F[:, 0], F[:, 1], F[:, 2])
        plt.show()

    if movie:

        fig = plt.figure()
        ax = plt.gca()
        entry_F = eval.data[0]['snapshot'][:, -2:]
        scat = plt.scatter(ref_points[:, 0], ref_points[:, 1])
        scat = plt.scatter(entry_F[:, 0], entry_F[:, 1])


        def update(frame_number):
            entry_F = eval.data[frame_number]['snapshot'][:,-2:]
            scat.set_offsets(entry_F)

            eps = 0.1
            min = np.min(entry_F, axis=0) - eps
            max = np.max(entry_F, axis=0) + eps
            ax.set_xlim(min[0], max[0])
            ax.set_ylim(min[1], max[1])


        ani = animation.FuncAnimation(fig, update, frames=range(100))


        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('im.mp4', writer=writer)







