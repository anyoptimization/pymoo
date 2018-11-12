import time

import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.indicators.igd import IGD
from pymoo.util.plotting import plot, animate
from pymop.problems.dtlz import DTLZ4, DTLZ2
from pymop.problems.tnk import TNK
from pymop.problems.zdt import ZDT4, ZDT3, ZDT1

if __name__ == '__main__':

    # load the problem instance

    problem = DTLZ2(n_var=40, n_obj=10)
    problem.n_pareto_points = 92

    problem = ZDT1()

    # create the algorithm instance by specifying the intended parameters
    from pymoo.algorithms.nsga3 import NSGA3

    algorithm = NSGA2(pop_size=100)

    for i in range(30):
        start_time = time.time()

        # number of generations to run it
        n_gen = 400

        # solve the problem and return the results
        X, F, G = algorithm.solve(problem,
                                  evaluator=(algorithm.pop_size * n_gen),
                                  seed=i,
                                  return_only_feasible=False,
                                  return_only_non_dominated=False,
                                  disp=False,
                                  save_history=False)

        print(IGD(problem.pareto_front()).calc(F))

        #print("--- %s seconds ---" % (time.time() - start_time))

    scatter_plot = False
    save_animation = False

    if scatter_plot:
        plot(F)

    if algorithm.save_history and save_animation:
        H = np.concatenate([e['pop'].F[None, :, :] for e in algorithm.history], axis=0)
        animate('%s.mp4' % problem.name(), H, problem)
