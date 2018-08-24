import time

import numpy as np

from pymoo.util.plotting import plot, animate

if __name__ == '__main__':

    from pymop.problems.zdt import ZDT1
    problem = ZDT1(n_var=30)

    start_time = time.time()

    from pymoo.optimize import minimize
    res = minimize(problem,
                   method='nsga3',
                   method_args={'pop_size': 92},
                   termination=('n_eval', 92 * 200),
                   seed=2,
                   save_history=True,
                   disp=True)

    print("--- %s seconds ---" % (time.time() - start_time))

    scatter_plot = True
    save_animation = True

    if scatter_plot:
        plot(res['F'])

    if save_animation:
        H = np.concatenate([e['pop'].F[None, :, :] for e in res['history']], axis=0)
        animate('%s.mp4' % problem.name(), H, problem)
