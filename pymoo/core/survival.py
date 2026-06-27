"""Survival selection strategies for filtering populations."""

from abc import abstractmethod
from typing import Any, Optional, Tuple

import numpy as np

from pymoo.core.population import Population
from pymoo.util import default_random_state


class Survival:
    """Base class for survival selection operators."""

    def __init__(self, filter_infeasible: bool = True) -> None:
        """Initialize survival operator.

        Args:
            filter_infeasible: Whether to separate feasible from infeasible solutions.
        """
        super().__init__()
        self.filter_infeasible = filter_infeasible

    @default_random_state
    def do(
        self,
        problem: Any,
        pop: Any,
        *args: Any,
        n_survive: Optional[int] = None,
        random_state: Any = None,
        return_indices: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Select survivors from population.

        Args:
            problem: Problem instance.
            pop: Population to filter.
            n_survive: Number of survivors to select.
            random_state: Random state for reproducibility.
            return_indices: Whether to return indices instead of individuals.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Surviving individuals or their indices.
        """
        # make sure the population has at least one individual
        if len(pop) == 0:
            return pop

        if n_survive is None:
            n_survive = len(pop)

        n_survive = min(n_survive, len(pop))

        # if the split should be done beforehand
        if self.filter_infeasible and problem.has_constraints():
            # split feasible and infeasible solutions
            feas, infeas = split_by_feasibility(pop, sort_infeas_by_cv=True)

            if len(feas) == 0:
                survivors = Population()
            else:
                survivors = self._do(
                    problem,
                    pop[feas],
                    *args,
                    n_survive=min(len(feas), n_survive),
                    random_state=random_state,
                    **kwargs,
                )

            # calculate how many individuals are still remaining to be filled up with infeasible ones
            n_remaining = n_survive - len(survivors)

            # if infeasible solutions needs to be added
            if n_remaining > 0:
                survivors = Population.merge(survivors, pop[infeas[:n_remaining]])

        else:
            survivors = self._do(
                problem,
                pop,
                *args,
                n_survive=n_survive,
                random_state=random_state,
                **kwargs,
            )

        if return_indices:
            h = {}  # noqa: I
            for k, ind in enumerate(pop):
                h[ind] = k
            return [h[survivor] for survivor in survivors]
        else:
            return survivors

    @abstractmethod
    def _do(
        self,
        problem: Any,
        pop: Any,
        *args: Any,
        n_survive: Optional[int] = None,
        random_state: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Perform survival selection (to be implemented by subclasses)."""


class ToReplacement(Survival):
    """Survival operator that replaces individuals with better offspring."""

    def __init__(self, survival: Survival) -> None:
        """Initialize replacement operator.

        Args:
            survival: Survival operator to use for ranking.
        """
        super().__init__(False)
        self.survival = survival

    def _do(  # type: ignore[override]
        self,
        problem: Any,
        pop: Any,
        off: Any,
        random_state: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Perform replacement."""
        merged = Population.merge(pop, off)
        indices = self.survival.do(
            problem,
            merged,
            n_survive=len(merged),
            return_indices=True,
            random_state=random_state,
            **kwargs,
        )
        merged.set("__rank__", indices)

        for k in range(len(pop)):
            if off[k].get("__rank__") < pop[k].get("__rank__"):
                pop[k] = off[k]

        return pop


def split_by_feasibility(
    pop: Any,
    sort_infeas_by_cv: bool = True,
    sort_feas_by_obj: bool = False,
    return_pop: bool = False,
) -> Tuple[Any, ...]:
    """Split population into feasible and infeasible individuals.

    Args:
        pop: Population to split.
        sort_infeas_by_cv: Whether to sort infeasible by constraint violation.
        sort_feas_by_obj: Whether to sort feasible by objective.
        return_pop: Whether to return populations instead of indices.

    Returns:
        Feasible and infeasible indices/populations.
    """
    F, CV, b = pop.get("F", "CV", "FEAS")

    feasible = np.where(b)[0]
    infeasible = np.where(~b)[0]

    if sort_infeas_by_cv:
        infeasible = infeasible[np.argsort(CV[infeasible, 0])]

    if sort_feas_by_obj:
        feasible = feasible[np.argsort(F[feasible, 0])]

    if not return_pop:
        return feasible, infeasible
    else:
        return feasible, infeasible, pop[feasible], pop[infeasible]
