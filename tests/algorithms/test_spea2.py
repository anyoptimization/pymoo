import numpy as np

from pymoo.algorithms.moo.spea2 import SPEA2Survival
from pymoo.core.population import Population


class NoConstraintsProblem:

    def has_constraints(self):
        return False


def test_spea2_truncation_keeps_extreme_solutions():
    F = np.array([
        [2.0, 0.0],
        [1.0, 1.0],
        [0.0, 2.0],
    ])

    pop = Population.new(F=F)
    survival = SPEA2Survival()
    survivors = survival.do(NoConstraintsProblem(), pop, n_survive=2)

    np.testing.assert_array_equal(
        survivors.get("F"),
        np.array([
            [2.0, 0.0],
            [0.0, 2.0],
        ])
    )


def test_spea2_truncation_preserves_extremes_after_multiple_removals():
    F = np.array([
        [3.0, 0.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [0.0, 3.0],
    ])

    pop = Population.new(F=F)
    survival = SPEA2Survival()
    survivors = survival.do(NoConstraintsProblem(), pop, n_survive=2)

    np.testing.assert_array_equal(
        survivors.get("F"),
        np.array([
            [3.0, 0.0],
            [0.0, 3.0],
        ])
    )
