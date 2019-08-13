import numpy as np

from pymoo.performance_indicator.gd import GD
from pymoo.performance_indicator.igd import IGD
from pymoo.performance_indicator.hv import Hypervolume

width = 12


def format_float(f):
    if f >= 10:
        return f"%.{width - 6}E" % f
    else:
        return f"%.{width - 2}f" % f


def pareto_front_if_possible(problem):
    try:
        return problem.pareto_front()
    except:
        return None


def disp_cv(CV):
    min_constr = format_float(np.min(CV))
    mean_constr = format_float(np.mean(CV))
    return "cv (min/avg)", f"{min_constr} / {mean_constr}", width * 2 + 3


def disp_single_objective(problem, evaluator, algorithm, pf=None):
    attrs = [('n_gen', algorithm.n_gen, 5),
             ('n_eval', evaluator.n_eval, 7)]

    F, CV, feasible = algorithm.pop.get("F", "CV", "feasible")
    feasible = np.where(feasible[:, 0])[0]

    if problem.n_constr > 0:
        attrs.append(disp_cv(CV))

    if len(feasible) > 0:
        attrs.extend([
            ('favg', format_float(np.mean(F[feasible])), width),
            ('fopt', format_float(np.min(F[feasible])), width)
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

    if isinstance(pf, bool):
        if pf:
            pf = pareto_front_if_possible(problem)
        else:
            pf = None

    if problem.n_constr > 0:
        attrs.append(disp_cv(CV))

    if len(feasible) > 0:
        if pf is not None:
            attrs.append(('igd', format_float(IGD(pf).calc(F[feasible])), width))
            attrs.append(('gd', format_float(GD(pf).calc(F[feasible])), width))
            if problem.n_obj == 2:
                attrs.append(('hv', format_float(Hypervolume(pf=pf).calc(F[feasible])), width))
    else:
        attrs.append(('igd', "-", width))
        attrs.append(('gd', "-", width))
        if problem.n_obj == 2:
            attrs.append(('hv', "-", width))

    return attrs
