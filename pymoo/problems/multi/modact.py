import os

import numpy as np

from pymoo.model.problem import Problem
from pymoo.problems.util import load_pareto_front_from_file


class MODAct(Problem):
    """Multi-Objective Design of Actuators

    MODAct is a framework for real-world constrained multi-objective optimization.
    Refer to the python package https://github.com/epfl-lamd/modact from requirements.

    Best-known Pareto fronts must be downloaded from here: https://doi.org/10.5281/zenodo.3824302

    Parameters
    ----------

    function: str or modact.problems
        The name of the benchmark problem to use either as a string or the
        problem object instance. Example values: cs1, cs3, ct2, ct4, cts3

    References:
    ----------
    C. Picard and J. Schiffmann, “Realistic Constrained Multi-Objective Optimization Benchmark Problems from Design,”
    IEEE Transactions on Evolutionary Computation, pp. 1–1, 2020.
    """
    def __init__(self, function, **kwargs):
        try:
            import modact.problems as pb
        except:
            raise Exception("Please install the modact library: https://github.com/epfl-lamd/modact")

        if isinstance(function, pb.Problem):
            self.fct = function
        else:
            self.fct = pb.get_problem(function)
        lb, ub = self.fct.bounds()
        n_var = len(lb)
        n_obj = len(self.fct.weights)
        n_constr = len(self.fct.c_weights)
        xl = lb
        xu = ub

        self.weights = np.array(self.fct.weights)
        self.c_weights = np.array(self.fct.c_weights)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl,
                         xu=xu, elementwise_evaluation=True, type_var=np.double,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        f, g = self.fct(x)
        out["F"] = np.array(f)*-1*self.weights
        out["G"] = np.array(g)*self.c_weights

    def _calc_pareto_front(self, *args, **kwargs):
        """Loads the corresponding PF if it exists"""
        fname = f"{self.fct.name}_PF.dat"
        F = load_pareto_front_from_file(os.path.join("modact", fname))
        if F is not None:
            return F*self.weights*-1
