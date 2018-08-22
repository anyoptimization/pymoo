import time

import numpy as np

from pymoo.util.plotting import plot, animate
from pymop.problems.dtlz import DTLZ4

if __name__ == '__main__':

    # load the problem instance
    from pymop.problems.zdt import ZDT3

    problem = DTLZ4(n_var=12, n_obj=3)
    problem.n_pareto_points = 92

    # create the algorithm instance by specifying the intended parameters
    from pymoo.algorithms.nsga3 import NSGA3

    algorithm = NSGA3(pop_size=92,
                 ref_dirs=None,
                 prob_cross=0.9,
                 eta_cross=20,
                 prob_mut=None,
                 eta_mut=30)

    start_time = time.time()

    # number of generations to run it
    n_gen = 500

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
