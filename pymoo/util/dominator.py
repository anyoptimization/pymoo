"""Utilities for determining dominance relations between individuals."""

from typing import Any

import numpy as np


def get_relation(ind_a: Any, ind_b: Any) -> int:
    """Get dominance relation between two individuals.

    Args:
        ind_a: First individual.
        ind_b: Second individual.

    Returns:
        1 if ind_a dominates ind_b, -1 if ind_b dominates ind_a, 0 if indifferent.
    """
    return Dominator.get_relation(ind_a.F, ind_b.F, ind_a.CV[0], ind_b.CV[0])


class Dominator:
    """Utility class for computing dominance relations."""

    @staticmethod
    def get_relation(a: Any, b: Any, cva: Any = None, cvb: Any = None) -> int:
        """Determine dominance relation between two objective vectors.

        Args:
            a: Objective vector of first solution.
            b: Objective vector of second solution.
            cva: Constraint violation of first solution.
            cvb: Constraint violation of second solution.

        Returns:
            1 if a dominates b, -1 if b dominates a, 0 if indifferent.
        """
        if cva is not None and cvb is not None:
            if cva < cvb:
                return 1
            elif cvb < cva:
                return -1

        val = 0
        for i in range(len(a)):
            if a[i] < b[i]:
                if val == -1:
                    return 0
                val = 1
            elif b[i] < a[i]:
                if val == 1:
                    return 0
                val = -1
        return val

    @staticmethod
    def calc_domination_matrix_loop(F: Any, G: Any) -> Any:
        """Calculate domination matrix using loop-based method.

        Args:
            F: Population objectives array.
            G: Population constraint violations array.

        Returns:
            Domination matrix where M[i,j] = 1 if i dominates j,
            -1 if j dominates i, 0 otherwise.
        """
        n = F.shape[0]
        CV = np.sum(G * (G > 0).astype(float), axis=1)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = Dominator.get_relation(F[i, :], F[j, :], CV[i], CV[j])
                M[j, i] = -M[i, j]

        return M

    @staticmethod
    def calc_domination_matrix(F: Any, _F: Any = None, epsilon: float = 0.0) -> Any:
        """Calculate domination matrix using vectorized method.

        Args:
            F: Population objectives array.
            _F: Reference front (if None, uses F).
            epsilon: Tolerance for domination relation.

        Returns:
            Domination matrix where M[i,j] = 1 if i dominates j,
            -1 if j dominates i, 0 otherwise.
        """
        if _F is None:
            _F = F

        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

        M = np.logical_and(smaller, np.logical_not(larger)) * 1 + np.logical_and(larger, np.logical_not(smaller)) * -1

        return M
