import time

import matplotlib.pyplot as plt
from pymoo.algorithms.NSGAIII import NSGAIII
from pymoo.model.evaluator import Evaluator
from mpl_toolkits.mplot3d import Axes3D

from pyop.problems.dtlz import DTLZ2

if __name__ == '__main__':

    problem = DTLZ2(n_var=10, n_obj=3)
    start_time = time.time()

    # run the algorithm
    # algorithm = SingleObjectiveGeneticAlgorithm("binary")
    algorithm = NSGAIII("real", pop_size=92, verbose=True)
    eval = Evaluator(92 * 200)
    X, F, G = algorithm.solve(problem, evaluator=eval, seed=1)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(F)


    plot = True
    is_2d = F.shape[1] == 2
    is_3d = F.shape[1] == 3

    if plot and is_2d:
        plt.scatter(F[:, 0], F[:, 1])
        plt.show()

    if plot and is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(F[:, 0], F[:, 1], F[:, 2])
        plt.show()
