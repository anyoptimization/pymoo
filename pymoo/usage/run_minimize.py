import time

import numpy as np

from pymoo.util.plotting import plot, animate
from pymop.problems.zdt import ZDT1


def run():

    # create the optimization problem
    problem = ZDT1()

    start_time = time.time()

    # solve the given problem using an optimization algorithm (here: nsga2)
    from pymoo.optimize import minimize
    res = minimize(problem,
                   method='nsga2',
                   method_args={'pop_size': 100},
                   termination=('n_gen', 200),
                   seed=1,
                   save_history=True,
                   disp=True)
    F = res['F']

    print("--- %s seconds ---" % (time.time() - start_time))

    scatter_plot = True
    save_animation = False

    if scatter_plot:
        plot(F, problem)

    if save_animation:
        H = np.concatenate([e['pop'].F[None, :, :] for e in res['history']], axis=0)
        animate('%s.mp4' % problem.name(), H, problem)


if __name__ == '__main__':
    run()