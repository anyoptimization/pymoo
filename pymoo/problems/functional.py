"""Functional problem class for wrapping objective and constraint functions."""

import numpy as np

from pymoo.core.problem import ElementwiseProblem


def func_return_none(*args, **kwargs):
    """Default function that returns None."""
    return None


class FunctionalProblem(ElementwiseProblem):
    """Problem defined by objective and constraint functions.

    Args:
        n_var: Number of variables.
        objs: Objective function or list of objective functions.
        constr_ieq: List of inequality constraint functions.
        constr_eq: List of equality constraint functions.
        func_pf: Function to compute Pareto front.
        func_ps: Function to compute Pareto set.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(
        self,
        n_var: int,
        objs,
        constr_ieq=None,
        constr_eq=None,
        func_pf=func_return_none,
        func_ps=func_return_none,
        **kwargs,
    ) -> None:
        if constr_ieq is None:
            constr_ieq = []
        if constr_eq is None:
            constr_eq = []

        # if only a single callable is provided (for single-objective problems) convert it to a list
        if callable(objs):
            objs = [objs]

        self.objs = objs
        self.constr_ieq = constr_ieq
        self.constr_eq = constr_eq
        self.func_pf = func_pf
        self.func_ps = func_ps

        super().__init__(
            n_var=n_var,
            n_obj=len(self.objs),
            n_ieq_constr=len(constr_ieq),
            n_eq_constr=len(constr_eq),
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objective and constraint functions.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        out["F"] = np.array([obj(x) for obj in self.objs])
        out["G"] = np.array([constr(x) for constr in self.constr_ieq])
        out["H"] = np.array([constr(x) for constr in self.constr_eq])

    def _calc_pareto_front(self, *args, **kwargs):
        """Calculate the Pareto front.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Pareto front or None.
        """
        return self.func_pf(*args, **kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        """Calculate the Pareto set.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Pareto set or None.
        """
        return self.func_ps(*args, **kwargs)
