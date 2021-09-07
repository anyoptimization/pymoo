import pytest

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.crossover import ElementwiseCrossover
from pymoo.factory import get_crossover, get_problem
from pymoo.operators.crossover.dex import DEX
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling, FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.single.traveling_salesman import create_random_tsp_problem


@pytest.mark.parametrize('name', ['real_de', 'real_sbx', 'real_pcx', 'real_exp'])
def test_crossover_real(name):
    crossover = get_crossover(name, prob=0.95)
    method = GA(pop_size=20, crossover=crossover)
    minimize(get_problem("sphere"), method, ("n_gen", 20))
    assert True


@pytest.mark.parametrize('name', ['bin_ux', 'bin_hux', 'bin_one_point', 'bin_two_point'])
def test_crossover_bin(name):
    crossover = get_crossover(name, prob=0.95)
    method = NSGA2(pop_size=20, crossover=crossover)
    minimize(get_problem("zdt5"), method, ("n_gen", 20))
    assert True


@pytest.mark.parametrize('name', ['perm_ox', 'perm_erx'])
def test_crossover_perm(name):
    crossover = get_crossover(name, prob=0.95)
    method = GA(pop_size=20, crossover=crossover, mutation=InversionMutation(), sampling=PermutationRandomSampling())
    minimize(create_random_tsp_problem(10), method, ("n_gen", 20))
    assert True


def test_elementwise_crossover():

    problem = get_problem('zdt1')

    sampling = FloatRandomSampling()

    pop = sampling.do(problem, n_samples=3)

    crossover = ElementwiseCrossover(SBX(20))
    off = crossover.do(problem, pop[0], pop[1])
    assert len(off) == 2

    crossover = ElementwiseCrossover(DEX())
    off = crossover.do(problem, pop[0], pop[1], pop[2])
    assert len(off) == 1