import numpy as np
import pytest

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_crossover, get_problem, get_mutation
from pymoo.operators.mutation.inversion import inversion_mutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.single.traveling_salesman import create_random_tsp_problem


@pytest.mark.parametrize('name', ['real_pm', 'none'])
def test_mutation_real(name):
    mut = get_mutation(name)
    method = GA(pop_size=20, mutation=mut)
    minimize(get_problem("sphere"), method, ("n_gen", 20))
    assert True


@pytest.mark.parametrize('name', ['bin_bitflip'])
def test_mutation_bin(name):
    mut = get_mutation(name)
    method = NSGA2(pop_size=20, crossover=get_crossover('bin_ux'), mutation=mut)
    minimize(get_problem("zdt5"), method, ("n_gen", 20))
    assert True


@pytest.mark.parametrize('name', ['perm_inv'])
def test_mutation_perm(name):
    mut = get_mutation(name, prob=0.95)
    method = GA(pop_size=20, crossover=get_crossover('perm_erx'), mutation=mut, sampling=PermutationRandomSampling())
    minimize(create_random_tsp_problem(10), method, ("n_gen", 20))
    assert True


def test_inversion_mutation():
    y = np.array([1, 2, 3, 4, 5])
    start = 1
    end = 3
    mut = inversion_mutation(y, seq=(start, end))
    np.testing.assert_allclose(mut, np.array([1, 4, 3, 2, 5]))
