import pytest

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize


@pytest.mark.parametrize('selection', ["rand", "best", "target-to-best"])
@pytest.mark.parametrize('crossover', ["bin", "exp"])
def test_de(selection, crossover):
    problem = get_problem("ackley", n_var=10)

    algorithm = DE(
        pop_size=100,
        sampling=LHS(),
        variant=f"DE/{selection}/1/{crossover}")

    ret = minimize(problem,
             algorithm,
             ('n_gen', 20),
             seed=1,
             verbose=True)

    assert len(ret.opt) > 0
