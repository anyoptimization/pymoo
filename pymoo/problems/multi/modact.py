"""Multi-objective design of actuators problem."""

import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.util.remote import Remote


class MODAct(ElementwiseProblem):
    """Multi-objective design of actuators.

    MODAct is a framework for real-world constrained multi-objective optimization.
    Refer to the python package https://github.com/epfl-lamd/modact from requirements.

    Best-known Pareto fronts must be downloaded from here:
    https://doi.org/10.5281/zenodo.3824302

    Parameters
    ----------
    function : str or modact.problems
        The name of the benchmark problem to use either as a string or the
        problem object instance. Example values: cs1, cs3, ct2, ct4, cts3
    """

    def __init__(
        self, function: str | object, pf: np.ndarray | None = None, **kwargs
    ) -> None:

        self.function: str | object = function
        self.pf: np.ndarray | None = pf

        try:
            import modact.problems as pb
        except ImportError as e:  # noqa: F841
            raise Exception(
                "Please install the modact library: https://github.com/epfl-lamd/modact"
            ) from e

        if isinstance(function, pb.Problem):
            self.fct = function
        else:
            self.fct = pb.get_problem(function)

        lb, ub = self.fct.bounds()
        n_var = len(lb)
        n_obj = len(self.fct.weights)
        n_ieq_constr = len(self.fct.c_weights)
        xl = lb
        xu = ub

        self.weights: np.ndarray = np.array(self.fct.weights)
        self.c_weights: np.ndarray = np.array(self.fct.c_weights)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
            vtype=float,
            **kwargs,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:  # type: ignore[override]  # noqa: ARG002
        f, g = self.fct(x)
        out["F"] = np.array(f) * -1 * self.weights
        out["G"] = np.array(g) * self.c_weights

    def _calc_pareto_front(self, *args, **kwargs):  # noqa: ARG002
        # allows to provide a custom pf - because of the size of files published by the author
        if self.pf is None:
            pf = Remote.get_instance().load(
                "pymoo", "pf", "MODACT", f"{self.function}.pf"
            )
            # pf = pf * [1, -1]
            pf = pf * self.weights * -1
            return pf
        else:
            return self.pf
