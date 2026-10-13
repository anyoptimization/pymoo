import numpy as np
import pytest

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.erx import ERX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.inversion import inversion_mutation, InversionMutation
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import PermutationRandomSampling, BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.single.traveling_salesman import create_random_tsp_problem


@pytest.mark.parametrize('mut', [PM()])
def test_mutation_real(mut):
    method = GA(pop_size=20, mutation=mut)
    minimize(get_problem("sphere"), method, ("n_gen", 20))
    assert True


@pytest.mark.parametrize('mut', [BitflipMutation()])
def test_mutation_bin(mut):
    method = NSGA2(pop_size=20, sampling=BinaryRandomSampling(), crossover=UX(), mutation=mut)
    minimize(get_problem("zdt5"), method, ("n_gen", 20))
    assert True


@pytest.mark.parametrize('mut', [InversionMutation()])
def test_mutation_perm(mut):
    method = GA(pop_size=20, crossover=ERX(), mutation=mut, sampling=PermutationRandomSampling())
    minimize(create_random_tsp_problem(10), method, ("n_gen", 20))
    assert True


def test_inversion_mutation():
    y = np.array([1, 2, 3, 4, 5])
    start = 1
    end = 3
    mut = inversion_mutation(y, seq=(start, end))
    np.testing.assert_allclose(mut, np.array([1, 4, 3, 2, 5]))


def test_bitflip_mutation_expectation():
    """Regression test: default BitflipMutation(prob=1.0) flips ~1 bit per chromosome.

    Uses a fixed random state to ensure the test is deterministic in CI.
    For n_var=100 and 400 individuals, the CLT gives a 99.9% confidence
    interval well within [0.75, 1.25].
    """
    from pymoo.core.problem import Problem
    from pymoo.core.population import Population

    class DummyBinaryProblem(Problem):
        def __init__(self, n_var=100):
            super().__init__(n_var=n_var, n_obj=1, xl=0, xu=1, vtype=bool)

    problem = DummyBinaryProblem(n_var=100)
    mut = BitflipMutation(prob=1.0)
    pop = Population.new("X", np.zeros((400, 100), dtype=bool))
    rng = np.random.default_rng(seed=42)
    mutated = mut.do(problem, pop, random_state=rng).get("X")
    mean_flips = np.mean(np.sum(mutated, axis=1))
    assert 0.75 <= mean_flips <= 1.25
