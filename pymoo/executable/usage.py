import time

import numpy as np

from pymoo.util.plotting import plot, animate
from pymop.problems.dtlz import DTLZ2


def run():
    # create the optimization problem
    problem = DTLZ2()

    start_time = time.time()

    # solve the given problem using an optimization algorithm (here: nsga2)
    from pymoo.optimize import minimize
    res = minimize(problem,
                   method='rnsga3',
                   method_args={'ref_points': np.array([[0.4, 0.1, 0.6], [0.8, 0.5, 0.8]]), 'pop_size': 182},
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
