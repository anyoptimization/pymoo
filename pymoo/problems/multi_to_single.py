"""Decomposition-based single-objective conversion of multi-objective problems."""

from typing import Any

from pymoo.core.meta import Meta
from pymoo.core.problem import Problem


class MultiToSingleObjective(Meta, Problem):
    """Convert a multi-objective problem to a single-objective one using decomposition.

    Args:
        problem: The underlying multi-objective problem.
        decomposition: The decomposition method to use.
        kwargs: Additional keyword arguments for the decomposition.
    """

    def __init__(
        self, problem: Any, decomposition: Any, kwargs: dict | None = None
    ) -> None:
        super().__init__(problem)
        self.decomposition = decomposition
        self.kwargs = kwargs if not None else dict()  # noqa: E712
        self.n_obj = 1

    def do(self, X: Any, return_values_of: Any, *args: Any, **kwargs: Any) -> dict:
        out = self.__object__.do(X, return_values_of, *args, **kwargs)
        F = out["F"]
        out["__F__"] = F
        out["F"] = self.decomposition.do(F, **self.kwargs)[:, None]
        return out
