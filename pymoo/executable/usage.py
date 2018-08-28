import time

import numpy as np

from pymoo.util.plotting import plot, animate
from pymop.problems.dtlz import DTLZ4, DTLZ2, DTLZ1
from pymop.problems.zdt import ZDT4, ZDT1


def run():
    problem = DTLZ1(n_var=12, n_obj=3)
    problem = ZDT4()

    start_time = time.time()

    from pymoo.optimize import minimize

    res = minimize(problem,
                   method='ansga3',
                   method_args={'pop_size': 91},
                   termination=('n_eval', 91 * 250),
                   seed=1,
                   save_history=True,
                   disp=False)
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
