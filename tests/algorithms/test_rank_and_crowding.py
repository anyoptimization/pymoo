import pytest

import numpy as np
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding, ConstrRankAndCrowding
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.functions import load_function


calc_mnn = load_function("calc_mnn")
calc_2nn = load_function("calc_2nn")
calc_pcd = load_function("calc_pcd")

# Get Python fallback versions for pytest
calc_mnn_python = load_function("calc_mnn", _type="python")
calc_2nn_python = load_function("calc_2nn", _type="python")


@pytest.mark.parametrize('crowding_func', ["mnn", "2nn", "cd", "pcd", "ce"])
@pytest.mark.parametrize('survival', [RankAndCrowding, ConstrRankAndCrowding])
def test_multi_run(crowding_func, survival):
    
    problem = get_problem("truss2d")

    NGEN = 250
    POPSIZE = 100
    SEED = 5
    
    nsga2 = NSGA2(pop_size=POPSIZE, survival=survival(crowding_func=crowding_func))

    res = minimize(problem,
                   nsga2,
                   ('n_gen', NGEN),
                   seed=SEED,
                   save_history=False,
                   verbose=False)
    
    assert len(res.opt) > 0


def test_cd_and_pcd():
    
    problem = get_problem("truss2d")

    NGEN = 200
    POPSIZE = 100
    SEED = 5
    
    nsga2 = NSGA2(pop_size=POPSIZE, survival=RankAndCrowding(crowding_func="pcd"))

    res = minimize(problem,
                   nsga2,
                   ('n_gen', NGEN),
                   seed=SEED,
                   save_history=False,
                   verbose=False)
    
    cd = calc_crowding_distance(res.F)
    pcd = calc_pcd(res.F)

    assert np.sum(np.abs(cd[~np.isinf(cd)] - pcd[~np.isinf(pcd)])) <= 1e-8
    
    new_F = res.F.copy()
    
    for j in range(10):
        
        cd = calc_crowding_distance(new_F)
        k = np.argmin(cd)
        new_F = new_F[np.arange(len(new_F)) != k]
    
    pcd = calc_pcd(res.F, n_remove=10)
    ind = np.argpartition(pcd, 10)[:10]
    
    new_F_alt = res.F.copy()[np.setdiff1d(np.arange(len(res.F)), ind)]
    
    assert np.sum(np.abs(new_F - new_F_alt)) <= 1e-8


def test_mnn():
    
    problem = get_problem("dtlz2")

    NGEN = 200
    POPSIZE = 100
    SEED = 5
    
    nsga2 = NSGA2(pop_size=POPSIZE, survival=RankAndCrowding(crowding_func="mnn"))

    res = minimize(problem,
                   nsga2,
                   ('n_gen', NGEN),
                   seed=SEED,
                   save_history=False,
                   verbose=False)
    
    surv_mnn = RankAndCrowding(crowding_func="mnn")
    surv_2nn = RankAndCrowding(crowding_func="2nn")
    
    surv_mnn_py = RankAndCrowding(crowding_func=calc_mnn_python)
    surv_2nn_py = RankAndCrowding(crowding_func=calc_2nn_python)
    
    random_state_1 = np.random.default_rng(12)
    pop_mnn = surv_mnn.do(problem, res.pop, n_survive=80, random_state=random_state_1)
    
    random_state_2 = np.random.default_rng(12)
    pop_mnn_py = surv_mnn_py.do(problem, res.pop, n_survive=80, random_state=random_state_2)
    
    assert np.sum(np.abs(pop_mnn.get("F") - pop_mnn_py.get("F"))) <= 1e-8
    
    random_state_3 = np.random.default_rng(12)
    pop_2nn = surv_2nn.do(problem, res.pop, n_survive=70, random_state=random_state_3)
    
    random_state_4 = np.random.default_rng(12)
    pop_2nn_py = surv_2nn_py.do(problem, res.pop, n_survive=70, random_state=random_state_4)
    
    assert np.sum(np.abs(pop_2nn.get("F") - pop_2nn_py.get("F"))) <= 1e-8