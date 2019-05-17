import numpy as np

from pymoo.indicators.gd import GD
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD


def disp_single_objective(problem, evaluator, algorithm, pf=None):
    attrs = [('n_gen', algorithm.n_gen, 5),
             ('n_eval', evaluator.n_eval, 7)]

    F, CV, feasible = algorithm.pop.get("F", "CV", "feasible")
    feasible = np.where(feasible[:, 0])[0]

    if problem.n_constr > 0:
        attrs.append(('cv (min/avg)', "%.5f / %.5f" % (np.min(CV), np.mean(CV)), 13))

    if len(feasible) > 0:
        attrs.extend([
            ('favg', "%.5f" % np.mean(F[feasible]), 5),
            ('fopt', "%.10f" % np.min(F[feasible]), 5)
        ])
    else:
        attrs.extend([
            ('favg', "-", 5),
            ('fopt', "-", 5)
        ])

    return attrs


def disp_multi_objective(problem, evaluator, algorithm, pf=None):
    attrs = [('n_gen', algorithm.n_gen, 5),
             ('n_eval', evaluator.n_eval, 7)]

    F, CV, feasible = algorithm.pop.get("F", "CV", "feasible")
    feasible = np.where(feasible[:, 0])[0]

    if problem.n_constr > 0:
        attrs.append(('cv (min/avg)', "%.5f / %.5f" % (np.min(CV), np.mean(CV)), 13))

    if len(feasible) > 0:
        if pf is not None:
            attrs.append(('igd', "%.5f" % IGD(pf).calc(F[feasible]), 8))
            attrs.append(('gd', "%.5f" % GD(pf).calc(F[feasible]), 8))
            if problem.n_obj == 2:
                attrs.append(('hv', "%.5f" % Hypervolume(pf=pf).calc(F[feasible]), 8))
    else:
        attrs.append(('igd', "-", 8))
        attrs.append(('gd', "-", 8))
        if problem.n_obj == 2:
            attrs.append(('hv', "-", 8))

    return attrs
