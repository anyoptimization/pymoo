import time

import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.indicators.igd import IGD
from pymoo.util.plotting import plot, animate
from pymop.problems.dtlz import DTLZ4, DTLZ2
from pymop.problems.osy import OSY
from pymop.problems.tnk import TNK
from pymop.problems.welded_beam import WeldedBeam
from pymop.problems.zdt import ZDT4, ZDT1

if __name__ == '__main__':

    # load the problem instance

    problem = DTLZ2(n_var=50, n_obj=30)
    problem.n_pareto_points = 92

    #problem = OSY()

    # create the algorithm instance by specifying the intended parameters
    from pymoo.algorithms.nsga3 import NSGA3

    algorithm = NSGA2(pop_size=200, eta_cross=5, eta_mut=5)

    start_time = time.time()

    # number of generations to run it
    n_gen = 250

    # solve the problem and return the results
    X, F, G = algorithm.solve(problem,
                              evaluator=(algorithm.pop_size * n_gen),
                              seed=23,
                              return_only_feasible=False,
                              return_only_non_dominated=False,
                              disp=True,
                              save_history=False)

    print("--- %s seconds ---" % (time.time() - start_time))

    scatter_plot = True
    save_animation = False

    if scatter_plot:
        plot(F)

    if algorithm.save_history and save_animation:
        H = np.concatenate([e['pop'].F[None, :, :] for e in algorithm.history], axis=0)
        animate('%s.mp4' % problem.name(), H, problem)
