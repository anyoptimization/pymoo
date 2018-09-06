import time

import numpy as np

from pymoo.util.plotting import plot, animate
from pymoo.util.reference_directions import get_ref_dirs_from_section
from pymop.problems.dtlz import DTLZ2
from pymop.problems.zdt import ZDT1, ZDT4, ZDT


def run():

    class ZDT1(ZDT):
        def __init__(self, n_var=30):
            ZDT.__init__(self, n_var)

        def _calc_pareto_front(self):
            x1 = np.arange(0, 1.01, 0.01)
            return np.array([x1, 1 - np.sqrt(x1)]).T

        def _evaluate(self, x, f, individuals):
            f[:, 0] = x[:, 0]
            g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
            f[:, 1] = g * (1 - np.power((f[:, 0] / g), 0.5))

    start_time = time.time()

    problem = DTLZ2()

    from pymoo.optimize import minimize

    res = minimize(problem,
                   method='rnsga3',
                   method_args={'ref_points': np.array([[0.4, 0.1, 0.6], [0.8,   0.5, 0.8]]), 'pop_per_ref_point' : 91},
                   #method_args={'pop_size': 100},
                   termination=('n_gen', 200),
                   seed=1,
                   save_history=True,
                   disp=True)

    X, F = res['X'], res['F']



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
