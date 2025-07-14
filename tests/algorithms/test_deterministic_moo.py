import numpy as np
import pytest

from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
# Multi-objective algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.kgb import KGB
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.pinsga2 import PINSGA2
from pymoo.algorithms.moo.moead import ParallelMOEAD

from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.pinsga2 import AutomatedDM


# Simple deterministic automated decision maker for testing
class SimpleDeterministicDM(AutomatedDM):
    def __init__(self, random_state=None):
        super().__init__()
        self.random_state = random_state
    
    def makeDecision(self, F):
        # Simple deterministic preference: prefer solutions with lower first objective
        return F[:, 0].argsort()
    
    def makePairwiseDecision(self, F):
        # For pairwise comparison, create deterministic rankings based on first objective
        ranks = self.makeDecision(F)
        return ranks


# Helper function to create algorithm instances with required parameters
def create_algorithm_instance(algorithm_class):
    """Create algorithm instance with proper parameters."""
    if algorithm_class in [NSGA3, UNSGA3, CTAEA, RVEA, ]:
        # These algorithms need reference directions - create them manually to avoid random_state issues
        from pymoo.util.ref_dirs.das_dennis import DasDennis
        ref_dirs_gen = DasDennis(n_partitions=12, n_dim=2)
        ref_dirs = ref_dirs_gen.next()
        return algorithm_class(ref_dirs=ref_dirs)
    elif algorithm_class == RNSGA2:
        # RNSGA2 needs reference points
        ref_points = np.array([[0.2, 0.2], [0.8, 0.8]])
        return algorithm_class(ref_points=ref_points)
    elif algorithm_class == RNSGA3:
        # RNSGA3 needs reference points and pop_per_ref_point
        ref_points = np.array([[0.2, 0.2], [0.8, 0.8]])
        return algorithm_class(ref_points=ref_points, pop_per_ref_point=10)
    elif algorithm_class in [MOEAD, ParallelMOEAD]:
        # MOEAD and ParallelMOEAD need reference directions - create them manually to avoid random_state issues
        from pymoo.util.ref_dirs.das_dennis import DasDennis
        ref_dirs_gen = DasDennis(n_partitions=12, n_dim=2)
        ref_dirs = ref_dirs_gen.next()
        return algorithm_class(ref_dirs=ref_dirs)
    elif algorithm_class == PINSGA2:
        # PINSGA2 needs an automated decision maker for non-interactive testing
        automated_dm = SimpleDeterministicDM()
        return algorithm_class(automated_dm=automated_dm)
    else:
        # Other algorithms can be instantiated without parameters
        return algorithm_class()


# Algorithm classes for testing
MULTI_OBJECTIVE_ALGORITHM_CLASSES = [
    NSGA2, RNSGA2, NSGA3, UNSGA3, RNSGA3, MOEAD, ParallelMOEAD,
    AGEMOEA, AGEMOEA2, CTAEA, SMSEMOA, RVEA, KGB, 
    SPEA2, DNSGA2, PINSGA2, MOPSO_CD, CMOPSO
]

# Test problem
MULTI_OBJECTIVE_PROBLEM = ZDT1()


@pytest.mark.parametrize('algorithm_class', MULTI_OBJECTIVE_ALGORITHM_CLASSES)
def test_multi_objective_deterministic(algorithm_class):
    """Test that multi-objective algorithms produce identical results with the same seed."""
    seed = 42
    n_gen = 10
    
    # Create fresh algorithm instances for each run using helper function
    algorithm1 = create_algorithm_instance(algorithm_class)
    algorithm2 = create_algorithm_instance(algorithm_class)
    
    # Run algorithm twice with same seed
    res1 = minimize(MULTI_OBJECTIVE_PROBLEM, algorithm1, ('n_gen', n_gen), seed=seed, verbose=False)
    res2 = minimize(MULTI_OBJECTIVE_PROBLEM, algorithm2, ('n_gen', n_gen), seed=seed, verbose=False)
    
    # Results should be identical
    np.testing.assert_allclose(res1.F, res2.F, rtol=1e-6, atol=1e-6,
                                  err_msg=f"Algorithm {algorithm_class.__name__} is not deterministic")
    np.testing.assert_allclose(res1.X, res2.X, rtol=1e-6, atol=1e-6,
                                  err_msg=f"Algorithm {algorithm_class.__name__} is not deterministic")


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results."""
    n_gen = 10
    
    results = []
    for seed in [1, 42, 123]:
        algorithm = create_algorithm_instance(NSGA2)
        res = minimize(MULTI_OBJECTIVE_PROBLEM, algorithm, ('n_gen', n_gen), seed=seed, verbose=False)
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