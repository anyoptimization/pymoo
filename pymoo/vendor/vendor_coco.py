import os

import numpy as np

from pymoo.core.problem import Problem
from pymoo.util.remote import Remote


class COCOProblem(Problem):

    def __init__(self, name, n_var=10, pf_from_file=True, **kwargs):
        self.function, self.instance, self.object = get_bbob(name, n_var)
        self.name = name
        self.pf_from_file = pf_from_file

        coco = self.object
        n_var, n_obj, n_ieq_constr = coco.number_of_variables, coco.number_of_objectives, coco.number_of_constraints
        xl, xu = coco.lower_bounds, coco.upper_bounds

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_ieq_constr=n_ieq_constr,
                         xl=xl,
                         xu=xu,
                         **kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        if self.n_obj == 1:
            fname = '._bbob_problem_best_parameter.txt'

            self.object._best_parameter(what="print")
            ps = np.loadtxt(fname)
            os.remove(fname)

            return ps

    def _calc_pareto_front(self, *args, **kwargs):
        if self.pf_from_file:
            return Remote.get_instance().load("pymoo", "pf", "bbob.pf", to="json")[str(self.function)][str(self.instance)]
        else:
            ps = self.pareto_set()
            if ps is not None:
                return self.evaluate(ps)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.object(x) for x in X])

    def __getstate__(self):
        d = self.__dict__.copy()
        d["object"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.object = get_bbob(self.name, self.n_var)


def get_bbob(name, n_var=10, **kwargs):
    try:
        import cocoex as ex
    except:
        raise Exception("COCO test suite not found. \nInstallation Guide: https://github.com/numbbo/coco")

    args = name.split("-")

    n_instance = int(args[-1])
    n_function = int(args[-2].replace("f", ""))

    assert 1 <= n_function <= 24, f"BBOB has 24 different functions to be chosen. {n_function} is out of range."

    suite_filter_options = f"function_indices: {n_function} " \
                           f"instance_indices: {n_instance} " \
                           f"dimensions: {n_var}"

    problems = ex.Suite("bbob", "", suite_filter_options)
    assert len(problems) == 1, "COCO problem not found."

    coco = problems.next_problem()

    return n_function, n_instance, coco

