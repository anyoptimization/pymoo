import time

import numpy as np

from pymoo.util.plotting import plot, animate
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop import DTLZ1


def run():
    # create the optimization problem
    problem = DTLZ1(n_var=8, n_obj=3)
    problem.n_pareto_points = 91

    start_time = time.time()

    ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()

    # solve the given problem using an optimization algorithm (here: nsga2)
    from pymoo.optimize import minimize
    res = minimize(problem,
                   method='nsga3',
                   method_args={
                       'ref_dirs': ref_dirs,
                    },
                   termination=('n_gen', 400),
                   seed=3,
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
