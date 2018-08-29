import numpy as np

from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD


def disp_single_objective(problem, evaluator, D):
    attrs = [('n_gen', D['n_gen'], 5),
             ('n_eval', evaluator.n_eval, 7)]

    if problem.n_constr > 0:
        attrs.append(('cv (min/avg)', "%.5f / %.5f" % (np.min(D['pop'].CV), np.mean(D['pop'].CV)), 13))

    attrs.extend([
             ('favg', "%.5f" % np.mean(D['pop'].F), 5),
             ('fopt', "%.10f" % np.min(D['pop'].F), 5)
             ])

    return attrs


def disp_multi_objective(problem, evaluator, D):
    attrs = [('n_gen', D['n_gen'], 5),
             ('n_eval', evaluator.n_eval, 7)]

    if problem.n_constr > 0:
        attrs.append(('cv (min/avg)', "%.5f / %.5f" % (np.min(D['pop'].CV), np.mean(D['pop'].CV)), 13))

    pf = problem.pareto_front()
    if pf is not None:
        attrs.append(('igd', "%.5f" % IGD(pf).calc(D['pop'].F), 8))
        attrs.append(('gd', "%.5f" % GD(pf).calc(D['pop'].F), 8))

    return attrs
