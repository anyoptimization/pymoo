import numpy as np

from pymoo.indicators.gd import GD
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD


def disp_single_objective(problem, evaluator, D, pf=None):
    attrs = [('n_gen', D['n_gen'], 5),
             ('n_eval', evaluator.n_eval, 7)]

    F, CV = D['pop'].get("F", "CV")

    if problem.n_constr > 0:
        attrs.append(('cv (min/avg)', "%.5f / %.5f" % (np.min(CV), np.mean(CV)), 13))

    attrs.extend([
        ('favg', "%.5f" % np.mean(F), 5),
        ('fopt', "%.10f" % np.min(F), 5)
    ])

    return attrs


def disp_multi_objective(problem, evaluator, D, pf=None):
    attrs = [('n_gen', D['n_gen'], 5),
             ('n_eval', evaluator.n_eval, 7)]

    F, CV = D['pop'].get("F", "CV")

    if problem.n_constr > 0:
        attrs.append(('cv (min/avg)', "%.5f / %.5f" % (np.min(CV), np.mean(CV)), 13))

    if pf is not None:
        attrs.append(('igd', "%.5f" % IGD(pf).calc(F), 8))
        attrs.append(('gd', "%.5f" % GD(pf).calc(F), 8))
        if problem.n_obj == 2:
            attrs.append(('hv', "%.5f" % Hypervolume(pf).calc(F), 8))

    return attrs
