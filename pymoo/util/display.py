from pymoo.indicators.igd import IGD
import numpy as np


def disp_single_objective(problem, evaluator, D):
    attrs = [('n_gen', D['n_gen'], 5),
             ('n_eval', evaluator.n_eval, 7),
             ('favg', "%.5f" % np.mean(D['pop'].F), 5),
             ('fopt', "%.10f" % np.min(D['pop'].F), 5)
             ]
    return attrs


def disp_multi_objective(problem, evaluator, D):
    attrs = [('n_gen', D['n_gen'], 5),
             ('n_eval', evaluator.n_eval, 7)]

    pf = problem.pareto_front()
    if pf is not None:
        attrs.append(('igd', "%.5f" % IGD(pf).calc(D['pop'].F), 8))

    return attrs
