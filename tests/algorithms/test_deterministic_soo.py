import numpy as np
import pytest

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES, SimpleCMAES, BIPOPCMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.direct import DIRECT
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.ga import GA, BGA
from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.nrbo import NRBO
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.pso_ep import EPPSO
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere


ALL_SINGLE_OBJECTIVE_ALGORITHMS = [
    GA, NicheGA, DE, ES, BRKGA, NelderMead, PatternSearch,
    CMAES, SimpleCMAES, BIPOPCMAES, ISRES, SRES, PSO, G3PCX, NRBO, 
    RandomSearch, DIRECT, EPPSO
]

# Test problem
SINGLE_OBJECTIVE_PROBLEM = Sphere(n_var=2)

@pytest.mark.parametrize('algorithm_class', ALL_SINGLE_OBJECTIVE_ALGORITHMS)
def test_single_objective_deterministic(algorithm_class):
    """Test that single-objective algorithms produce identical results with the same seed."""
    seed = 42
    n_gen = 10
    
    # Create fresh algorithm instances for each run
    algorithm1 = algorithm_class()
    algorithm2 = algorithm_class()
    
    # Run algorithm twice with same seed
    res1 = minimize(SINGLE_OBJECTIVE_PROBLEM, algorithm1, ('n_gen', n_gen), seed=seed, verbose=False)
    res2 = minimize(SINGLE_OBJECTIVE_PROBLEM, algorithm2, ('n_gen', n_gen), seed=seed, verbose=False)
    
    # Results should be identical
    np.testing.assert_allclose(res1.F, res2.F, rtol=1e-6, atol=1e-6,
                                  err_msg=f"Algorithm {algorithm_class.__name__} is not deterministic")
    np.testing.assert_allclose(res1.X, res2.X, rtol=1e-6, atol=1e-6,
                                  err_msg=f"Algorithm {algorithm_class.__name__} is not deterministic")


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results."""
    algorithm = DE()
    n_gen = 10
    
    results = []
    for seed in [1, 42, 123]:
        res = minimize(SINGLE_OBJECTIVE_PROBLEM, algorithm, ('n_gen', n_gen), seed=seed, verbose=False)
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


