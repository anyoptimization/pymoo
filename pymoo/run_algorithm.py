import time

import numpy as np

from pymoo.util.plotting import plot, animate

if __name__ == '__main__':

    # load the problem instance
    from pymop.problems.zdt import ZDT1
    problem = ZDT1(n_var=30)

    # create the algorithm instance by specifying the intended parameters
    from pymoo.algorithms.moead import MOEAD
    algorithm = MOEAD(pop_size=100)

    start_time = time.time()

    # number of generations to run it
    n_gen = 200

    # solve the problem and return the results
    X, F, G = algorithm.solve(problem,
                              evaluator=(100 * n_gen),
                              seed=2,
                              return_only_feasible=False,
                              return_only_non_dominated=False,
                              disp=True,
                              save_history=True)

    print("--- %s seconds ---" % (time.time() - start_time))

    scatter_plot = True
    save_animation = True

    if scatter_plot:
        plot(F)

    if save_animation:
        H = np.concatenate([e['pop'].F[None, :, :] for e in algorithm.history], axis=0)
        animate('%s.mp4' % problem.name(), H, problem)
