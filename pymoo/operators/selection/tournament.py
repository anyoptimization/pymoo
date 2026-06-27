"""Tournament selection operator."""

import math

import numpy as np

from pymoo.core.selection import Selection
from pymoo.util.misc import random_permutations
from pymoo.util import default_random_state


class TournamentSelection(Selection):
    """Tournament selection operator for genetic algorithms.

    Simulates a tournament between individuals. The selection pressure balances
    the greediness of the genetic algorithm.
    """

    def __init__(self, func_comp=None, pressure: int = 2, **kwargs) -> None:
        """Initialize the tournament selection operator.

        Args:
            func_comp: Comparison function with signature comp(pop, indices).
                Returns the index of the winner. If None, population is assumed
                to be sorted by criterion and indices are compared directly.
            pressure: Selection pressure (tournament size). Default is 2 for binary tournament.
            **kwargs: Additional keyword arguments.

        Raises:
            Exception: If func_comp is not provided.
        """
        super().__init__(**kwargs)

        # selection pressure to be applied
        self.pressure = pressure

        self.func_comp = func_comp
        if self.func_comp is None:
            raise Exception(
                "Please provide the comparing function for the tournament selection!"
            )

    def _do(  # type: ignore[override]
        self, _, pop, n_select: int, n_parents: int = 1, random_state=None, **kwargs
    ) -> np.ndarray:
        """Select parents using tournament selection.

        Args:
            _: Optimization problem (unused).
            pop: Population.
            n_select: Number of selections.
            n_parents: Number of parents per selection.
            random_state: Random state for reproducibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Index matrix of shape (n_select, n_parents).
        """
        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        P = random_permutations(n_perms, len(pop), random_state=random_state)[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        # compare using tournament function
        S = self.func_comp(pop, P, random_state=random_state, **kwargs)

        return np.reshape(S, (n_select, n_parents))


@default_random_state
def compare(
    a,
    a_val,
    b,
    b_val,
    method: str,
    return_random_if_equal: bool = False,
    random_state=None,
):
    """Compare two individuals based on their values.

    Args:
        a: First individual index.
        a_val: Value of first individual.
        b: Second individual index.
        b_val: Value of second individual.
        method: Comparison method ('larger_is_better' or 'smaller_is_better').
        return_random_if_equal: Whether to return random choice if values are equal.
        random_state: Random state for reproducibility.

    Returns:
        Index of the winning individual, or None if values are equal and
        return_random_if_equal is False.

    Raises:
        Exception: If method is not recognized.
    """
    if method == "larger_is_better":
        if a_val > b_val:
            return a
        elif a_val < b_val:
            return b
        else:
            if return_random_if_equal:
                return random_state.choice([a, b])
            else:
                return None
    elif method == "smaller_is_better":
        if a_val < b_val:
            return a
        elif a_val > b_val:
            return b
        else:
            if return_random_if_equal:
                return random_state.choice([a, b])
            else:
                return None
    else:
        msg = "Unknown method."
        raise ValueError(msg)
