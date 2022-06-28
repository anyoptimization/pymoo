import pytest

from pymoo.operators.crossover.erx import ERX
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.crossover.spx import SPX

from pymoo.operators.crossover.dex import DEX

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.single.traveling_salesman import create_random_tsp_problem


@pytest.mark.parametrize('crossover', [DEX(), SBX()])
def test_crossover_real(crossover):
    method = GA(pop_size=20, crossover=crossover)
    minimize(get_problem("sphere"), method, ("n_gen", 20))
    assert True


@pytest.mark.parametrize('crossover', [UX(), HUX(), SPX(), TwoPointCrossover()])
def test_crossover_bin(crossover):
    method = NSGA2(pop_size=20, crossover=crossover)
    minimize(get_problem("zdt5"), method, ("n_gen", 20))
    assert True


@pytest.mark.parametrize('crossover', [OrderCrossover(), ERX()])
def test_crossover_perm(crossover):
    method = GA(pop_size=20, crossover=crossover, mutation=InversionMutation(), sampling=PermutationRandomSampling())
    minimize(create_random_tsp_problem(10), method, ("n_gen", 20))
    assert True


