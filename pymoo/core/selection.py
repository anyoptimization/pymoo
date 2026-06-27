"""Selection operators for choosing individuals from a population."""

from abc import abstractmethod
from typing import Any

import numpy as np

from pymoo.core.operator import Operator
from pymoo.util import default_random_state


class Selection(Operator):
    """Base class for selection operators."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize selection operator."""
        super().__init__(**kwargs)

    @default_random_state
    def do(
        self,
        problem: Any,
        pop: Any,
        n_select: int,
        n_parents: int,
        to_pop: bool = True,
        *args: Any,
        random_state: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Choose individuals from population for reproduction.

        Args:
            problem: Problem instance providing bounds and constraints.
            pop: Population to select from.
            n_select: Number of individuals to select.
            n_parents: Number of parents needed per offspring.
            to_pop: Whether to convert indices to individuals.
            random_state: Random state for reproducibility.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Selected parents or indices.
        """
        ret = self._do(
            problem,
            pop,
            n_select,
            n_parents,
            *args,
            random_state=random_state,
            **kwargs,
        )

        # if some selections return indices they are used to create the individual list
        if (
            to_pop
            and isinstance(ret, np.ndarray)
            and np.issubdtype(ret.dtype, np.integer)
        ):
            ret = pop[ret]  # noqa: E501

        return ret

    @abstractmethod
    def _do(  # type: ignore[override]
        self,
        problem: Any,
        pop: Any,
        n_select: int,
        n_parents: int,
        *args: Any,
        random_state: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Perform selection (to be implemented by subclasses)."""
