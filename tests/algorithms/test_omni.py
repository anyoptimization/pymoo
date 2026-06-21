import numpy as np
import pytest

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.omni import (
    BOUNDARY,
    LooseDominator,
    NeighborBasedTournamentSelection,
    OmniOptimizer,
    calc_crowding_distance_in_space,
    calc_omni_crowding_distance,
)
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems.multi.omnitest import OmniTest


# ---------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------

# the omni-test problem has 3 Pareto sub-set bands per variable; this maps a solution to the
# band it belongs to so that the number of distinct Pareto subsets covered can be counted
_BANDS = [(1.0, 1.5), (3.0, 3.5), (5.0, 5.5)]


def _n_pareto_subsets_covered(X, tol=0.3):
    cells = set()
    for x in X:
        idx = []
        for v in x:
            band = -1
            for b, (lo, hi) in enumerate(_BANDS):
                if lo - tol <= v <= hi + tol:
                    band = b
            idx.append(band)
        if -1 not in idx:
            cells.add(tuple(idx))
    return len(cells)


# ---------------------------------------------------------------------------------------------------------
# Loose (dynamic epsilon) dominance
# ---------------------------------------------------------------------------------------------------------


def test_loose_dominator_epsilon_merges_close_solutions():
    # objective ranges are 1.0 in both objectives, so epsilon = delta * 1.0
    F = np.array([[0.0, 0.0], [0.05, 0.05], [1.0, 1.0]])

    # with delta=0 the loose dominance is the usual Pareto dominance: 0 dominates 1
    M = LooseDominator(delta=0.0).calc_domination_matrix(F)
    assert M[0, 1] == 1
    assert M[1, 0] == -1

    # with delta=0.1 (epsilon=0.1) solution 0 and 1 are closer than the margin -> same front
    M = LooseDominator(delta=0.1).calc_domination_matrix(F)
    assert M[0, 1] == 0
    assert M[1, 0] == 0

    # both still clearly dominate the distant solution 2
    assert M[0, 2] == 1
    assert M[1, 2] == 1


def test_loose_dominator_matrix_is_antisymmetric():
    rng = np.random.default_rng(1)
    F = rng.random((20, 2))
    M = LooseDominator(delta=0.01).calc_domination_matrix(F)
    np.testing.assert_array_equal(M, -M.T)


# ---------------------------------------------------------------------------------------------------------
# Crowding distance in objective and variable space
# ---------------------------------------------------------------------------------------------------------


def test_crowding_distance_objective_space():
    Y = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    cd = calc_crowding_distance_in_space(Y, space="objective")
    # the two extremes are boundary solutions, the middle gets the averaged normalized gap
    np.testing.assert_allclose(cd, [BOUNDARY, 1.0, BOUNDARY])


def test_crowding_distance_variable_space_has_no_infinite_boundaries():
    Y = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    cd = calc_crowding_distance_in_space(Y, space="variable")
    # in variable space no solution is treated as infinitely important
    assert np.all(cd < BOUNDARY)
    np.testing.assert_allclose(cd, [1.0, 1.0, 1.0])


def test_crowding_distance_small_front_all_boundary():
    for n in (1, 2):
        Y = np.arange(2 * n, dtype=float).reshape(n, 2)
        np.testing.assert_allclose(calc_crowding_distance_in_space(Y, space="objective"),
                                   np.full(n, BOUNDARY))


def test_combined_crowding_distance():
    F = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    X = F.copy()
    cd = calc_omni_crowding_distance(F, X)
    # objective space gives [BOUNDARY, 1, BOUNDARY], variable space [1, 1, 1];
    # every solution is above the (tiny) objective-space average so the max is taken
    np.testing.assert_allclose(cd, [BOUNDARY, 1.0, BOUNDARY])


def test_combined_crowding_distance_single_space():
    F = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    X = np.array([[0.0, 0.0], [3.0, 3.0], [6.0, 6.0]])

    only_obj = calc_omni_crowding_distance(F, X, obj_crowding=True, var_crowding=False)
    np.testing.assert_allclose(only_obj, calc_crowding_distance_in_space(F, space="objective"))

    only_var = calc_omni_crowding_distance(F, X, obj_crowding=False, var_crowding=True)
    np.testing.assert_allclose(only_var, calc_crowding_distance_in_space(X, space="variable"))


def test_combined_crowding_requires_a_space():
    with pytest.raises(ValueError):
        calc_omni_crowding_distance(np.zeros((3, 2)), np.zeros((3, 2)),
                                    obj_crowding=False, var_crowding=False)


# ---------------------------------------------------------------------------------------------------------
# Neighbor based tournament selection
# ---------------------------------------------------------------------------------------------------------


def test_neighbor_selection_shape_and_indices():
    problem = OmniTest(n_var=2)
    algorithm = OmniOptimizer(pop_size=20)
    algorithm.setup(problem, termination=("n_gen", 1), seed=1)
    algorithm.next()

    selection = NeighborBasedTournamentSelection()
    n_select, n_parents = 10, 2
    parents = selection.do(problem, algorithm.pop, n_select, n_parents,
                           algorithm=algorithm, random_state=algorithm.random_state, to_pop=False)

    assert parents.shape == (n_select, n_parents)
    assert parents.min() >= 0 and parents.max() < len(algorithm.pop)


# ---------------------------------------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------------------------------------


def test_omni_converges_on_omni_test():
    problem = OmniTest(n_var=2)
    res = minimize(problem, OmniOptimizer(pop_size=100), ("n_gen", 150), seed=1, verbose=False)

    assert len(res.opt) > 0
    assert IGD(problem.pareto_front()).do(res.F) < 0.05


def test_omni_maintains_multiple_pareto_subsets():
    problem = OmniTest(n_var=2)  # 3 ** 2 = 9 equivalent Pareto subsets

    omni = minimize(problem, OmniOptimizer(pop_size=100), ("n_gen", 150), seed=1, verbose=False)
    nsga2 = minimize(problem, NSGA2(pop_size=100), ("n_gen", 150), seed=1, verbose=False)

    omni_covered = _n_pareto_subsets_covered(omni.X)
    nsga2_covered = _n_pareto_subsets_covered(nsga2.X)

    # the omni-optimizer should maintain (almost) all equivalent subsets and never fewer
    # than NSGA-II, which has no variable-space niching
    assert omni_covered >= 8
    assert omni_covered >= nsga2_covered


def test_omni_is_deterministic():
    problem = OmniTest(n_var=2)
    res1 = minimize(problem, OmniOptimizer(pop_size=40), ("n_gen", 20), seed=42, verbose=False)
    res2 = minimize(problem, OmniOptimizer(pop_size=40), ("n_gen", 20), seed=42, verbose=False)
    np.testing.assert_allclose(res1.F, res2.F)
    np.testing.assert_allclose(res1.X, res2.X)
