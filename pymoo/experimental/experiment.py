import time

import numpy as np

from pymoo.util.plotting import plot, animate
from pymoo.util.reference_directions import get_ref_dirs_from_section
from pymop.problems.dtlz import DTLZ2
from pymop.problems.zdt import ZDT1, ZDT4


def run():
    problem = DTLZ2()
    # problem = ZDT1()

    start_time = time.time()

    from pymoo.optimize import minimize

    res = minimize(problem,
                   method='moead',
                   method_args={'ref_dirs': get_ref_dirs_from_section(problem.n_obj, 12)},
                   termination=('n_gen', 200),
                   seed=1,
                   save_history=True,
                   disp=True)
    X, F = res['X'], res['F']
    print(X)
    print(F)

    print("--- %s seconds ---" % (time.time() - start_time))

    scatter_plot = True
    save_animation = False

    if scatter_plot:
        plot(F, problem)

    if save_animation:
        def fun(ax, entry):
            nadir_point = entry['nadir_point'][None, :]
            ax.scatter(nadir_point[:, 0], nadir_point[:, 1], marker="x")

            F = entry['pop'].F

            # get the bounds for plotting and add padding
            min = np.min(np.concatenate([F, nadir_point], axis=0), axis=0)
            max = np.max(np.concatenate([F, nadir_point], axis=0), axis=0)

            min -= min * 0.1
            max += max * 0.1

            # set the scatter object with padding
            ax.set_xlim(min[0], max[0])
            ax.set_ylim(min[1], max[1])

        animate('%s.mp4' % problem.name(), res['history'], problem)


if __name__ == '__main__':
    run()
