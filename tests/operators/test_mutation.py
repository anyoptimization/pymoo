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
