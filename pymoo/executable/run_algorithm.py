import time

import numpy as np

from pymoo.util.plotting import plot, animate

if __name__ == '__main__':

    from pymoo.cpp import hello_module
    hello_module.print_hello_world()


    # load the problem instance
    from pymop.problems.zdt import ZDT3

    problem = ZDT3()

    # create the algorithm instance by specifying the intended parameters
    from pymoo.algorithms.nsga2 import NSGA2

    algorithm = NSGA2()

    start_time = time.time()

    # number of generations to run it
    n_gen = 200

    # solve the problem and return the results
    X, F, G = algorithm.solve(problem,
                              evaluator=(algorithm.pop_size * n_gen),
                              seed=15,
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
