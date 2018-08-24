import time

import numpy as np

from pymoo.algorithms.so_genetic_algorithm import SingleObjectiveGeneticAlgorithm
from pymoo.util.plotting import plot, animate
from pymop.problems.dtlz import DTLZ4
from pymop.problems.osy import OSY
from pymop.problems.tnk import TNK

if __name__ == '__main__':

    # load the problem instance
    from pymop.problems.zdt import ZDT3, ZDT4

    problem = DTLZ4(n_var=12, n_obj=3)
    problem.n_pareto_points = 92#

    problem = ZDT4()

    # create the algorithm instance by specifying the intended parameters
    from pymoo.algorithms.unsga3 import UNSGA3
    from pymoo.algorithms.nsga3 import NSGA3
    from pymoo.algorithms.nsga2 import NSGA2

    SingleObjectiveGeneticAlgorithm()

    algorithm = NSGA3(pop_size=100,
                      ref_dirs=None,
                      prob_cross=0.9,
                      eta_cross=20,
                      prob_mut=None,
                      eta_mut=15)

    start_time = time.time()

    # number of generations to run it
    n_gen = 250

    # solve the problem and return the results
    X, F, G = algorithm.solve(problem,
                              evaluator=(algorithm.pop_size * n_gen),
                              seed=25,
                              return_only_feasible=False,
                              return_only_non_dominated=False,
                              disp=True,
                              save_history=False)

    print("--- %s seconds ---" % (time.time() - start_time))

    scatter_plot = True
    save_animation = True

    if scatter_plot:
        plot(F)

    if algorithm.save_history and save_animation:
        H = np.concatenate([e['pop'].F[None, :, :] for e in algorithm.history], axis=0)
        animate('%s.mp4' % problem.name(), H, problem)
