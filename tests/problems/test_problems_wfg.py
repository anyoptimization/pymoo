import numpy as np
import pytest

from pymoo.problems import get_problem
from pymoo.problems.many import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from tests.problems.test_correctness import load


@pytest.mark.parametrize('name', [f"wfg{k}" for k in range(1, 10)])
@pytest.mark.parametrize('params', [(2, 6, 4), (3, 6, 4), (10, 20, 18)])
def test_problems(name, params):
    n_obj, n_var, k = params
    problem = get_problem(name, n_var, n_obj, k)

    X, F = load("problems", "WFG", f"{n_obj}obj", name.upper(), attrs=["x", "f"])
    _F = problem.evaluate(X)

    np.testing.assert_allclose(_F, F)

    # this is not tested automatically
    try:
        optprobs_F = []
        for x in X:
            from optproblems.base import Individual
            ind = Individual(phenome=x)
            from_optproblems(problem).evaluate(ind)
            optprobs_F.append(ind.objective_values)
        optprobs_F = np.array(optprobs_F)

        np.testing.assert_allclose(_F, optprobs_F)
    except:
        print("NOT MATCHING")



def from_optproblems(wfg):
    from optproblems.wfg import WFG1 as WFG1opt
    from optproblems.wfg import WFG2 as WFG2opt
    from optproblems.wfg import WFG3 as WFG3opt
    from optproblems.wfg import WFG4 as WFG4opt
    from optproblems.wfg import WFG5 as WFG5opt
    from optproblems.wfg import WFG6 as WFG6opt
    from optproblems.wfg import WFG7 as WFG7opt
    from optproblems.wfg import WFG8 as WFG8opt
    from optproblems.wfg import WFG9 as WFG9opt

    if isinstance(wfg, WFG1):
        return WFG1opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG2):
        return WFG2opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG3):
        return WFG3opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG4):
        return WFG4opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG5):
        return WFG5opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG6):
        return WFG6opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG7):
        return WFG7opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG8):
        return WFG8opt(wfg.n_obj, wfg.n_var, wfg.k)
    elif isinstance(wfg, WFG9):
        return WFG9opt(wfg.n_obj, wfg.n_var, wfg.k)


