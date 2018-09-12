import time

import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.model.termination import MaximumGenerationTermination
from pymoo.util.plotting import plot, animate, plot_3d
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.problem import ScaledProblem, ConvexProblem
from pymop.problems.dtlz import DTLZ2, DTLZ1

import matplotlib.pyplot as plt


def run():
    start_time = time.time()

    problem = ScaledProblem(DTLZ2(n_var=12, n_obj=3), 10)

    #plot_3d(problem.pareto_front())
    #plt.xlabel("X")
    #plt.ylabel("Y")
    #plt.zlabel("Z")
    #plt.show()

    algorithm = NSGA3(ref_dirs=UniformReferenceDirectionFactory(n_dim=3, n_partitions=12).do())

    res = algorithm.solve(problem,
                          termination=MaximumGenerationTermination(250),
                          seed=2,
                          save_history=True,
                          disp=True)

    X, F, history = res['X'], res['F'], res['history']

    print("--- %s seconds ---" % (time.time() - start_time))

    F = (F - problem.ideal_point()) / (problem.nadir_point() - problem.ideal_point())

    scatter_plot = True
    save_animation = False

    if scatter_plot:
        plot(F, problem)

    if save_animation:
        animate('%s.mp4' % problem.name(), history, problem)

    ideal_point = np.zeros(len(history))
    nadir_point = np.zeros(len(history))

    for i, entry in enumerate(history):
        ideal_point[i] = np.linalg.norm(entry.survival.ideal_point)
        nadir_point[i] = np.linalg.norm(entry.survival.intercepts + entry.survival.ideal_point)

    # plt.figure()
    # plt.plot(np.arange(len(history)), ideal_point)
    # plt.plot(np.arange(len(history)), nadir_point)
    # plt.show()

    print("test")


if __name__ == '__main__':
    run()
