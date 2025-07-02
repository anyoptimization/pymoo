import numpy as np
import pytest

from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.optimize import minimize


class MixedVariableProblem(ElementwiseProblem):
    """Test mixed variable problem for deterministic testing."""

    def __init__(self, **kwargs):
        vars = {
            "b": Binary(),
            "x": Choice(options=["nothing", "multiply"]),
            "y": Integer(bounds=(0, 2)),
            "z": Real(bounds=(0, 5)),
        }
        super().__init__(vars=vars, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        b, x, z, y = X["b"], X["x"], X["z"], X["y"]

        f = z + y
        if b:
            f = 100 * f

        if x == "multiply":
            f = 10 * f

        out["F"] = f


# All mixed variable algorithms that support deterministic behavior
ALL_MIXED_VARIABLE_ALGORITHMS = [
    MixedVariableGA,
    Optuna
]

# Test problem
MIXED_VARIABLE_PROBLEM = MixedVariableProblem()


@pytest.mark.parametrize('algorithm_class', ALL_MIXED_VARIABLE_ALGORITHMS)
def test_mixed_variable_deterministic(algorithm_class):
    """Test that mixed variable algorithms produce identical results with the same seed."""
    seed = 42
    n_evals = 50  # Use evaluations instead of generations for mixed variable problems
    
    # Create fresh algorithm instances for each run
    algorithm1 = algorithm_class()
    algorithm2 = algorithm_class()
    
    # Run algorithm twice with same seed
    res1 = minimize(MIXED_VARIABLE_PROBLEM, algorithm1, ('n_evals', n_evals), seed=seed, verbose=False)
    res2 = minimize(MIXED_VARIABLE_PROBLEM, algorithm2, ('n_evals', n_evals), seed=seed, verbose=False)
    
    # Results should be identical
    np.testing.assert_allclose(res1.F, res2.F, rtol=1e-6, atol=1e-6,
                                  err_msg=f"Algorithm {algorithm_class.__name__} is not deterministic")
    
    # For mixed variables, we need to compare the X values appropriately
    # Since X is a dictionary, we compare each variable type
    for key in res1.X.keys():
        if key in ["b", "x"]:  # Binary and choice variables should be exactly equal
            assert res1.X[key] == res2.X[key], f"Algorithm {algorithm_class.__name__} is not deterministic for variable {key}"
        else:  # Real and integer variables should be close
            np.testing.assert_allclose(res1.X[key], res2.X[key], rtol=1e-6, atol=1e-6,
                                         err_msg=f"Algorithm {algorithm_class.__name__} is not deterministic for variable {key}")


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results for mixed variable optimization."""
    algorithm = MixedVariableGA()
    n_evals = 50
    
    results = []
    for seed in [1, 42, 123]:
        res = minimize(MIXED_VARIABLE_PROBLEM, algorithm, ('n_evals', n_evals), seed=seed, verbose=False)
        results.append(res)
    
    # At least one pair should be different
    different_found = False
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            if not np.array_equal(results[i].F, results[j].F):
                different_found = True
                break
        if different_found:
            break
    
    assert different_found, "Different seeds should produce different results"