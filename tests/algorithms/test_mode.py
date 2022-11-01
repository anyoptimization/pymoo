import pytest

import numpy as np

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoo.algorithms.moo.nsder import NSDER
from pymoo.algorithms.moo.nsde import NSDE
from pymoo.algorithms.moo.gde3 import GDE3
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding, ConstrRankAndCrowding
from pymoo.util.ref_dirs import get_reference_directions


@pytest.mark.parametrize('survival', [RankAndCrowding, ConstrRankAndCrowding])
@pytest.mark.parametrize('crowding_func', ["mnn", "2nn", "cd", "pcd", "ce"])
def test_multi_run(survival, crowding_func):
    
    problem = get_problem("truss2d")

    NGEN = 250
    POPSIZE = 100
    SEED = 5
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back",
                survival=survival(crowding_func=crowding_func))

    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    assert len(res_gde3.opt) > 0
    

def test_multi_perf():
    
    problem = get_problem("truss2d")
    igd = IGD(pf=problem.pareto_front(), zero_to_one=True)
    
    NGEN = 250
    POPSIZE = 100
    SEED = 5
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back",
                survival=RankAndCrowding(crowding_func="cd"))

    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_gde3 = igd.do(res_gde3.F)
    assert igd_gde3 <= 0.08
    
    gde3p = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back",
                survival=RankAndCrowding(crowding_func="pcd"))

    res_gde3p = minimize(problem,
                        gde3p,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_gde3p = igd.do(res_gde3p.F)
    assert igd_gde3p <= 0.01
    
    nsde = NSDE(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back",
                survival=RankAndCrowding(crowding_func="pcd"))
        
    res_nsde = minimize(problem,
                        nsde,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_nsde = igd.do(res_nsde.F)
    assert igd_nsde <= 0.01

@pytest.mark.parametrize('selection', ["rand", "current-to-rand", "ranked"])
@pytest.mark.parametrize('crossover', ["bin", "exp"])
@pytest.mark.parametrize('crowding_func', ["mnn", "2nn"])
def test_many_run(selection, crossover, crowding_func):
    
    problem = get_problem("dtlz2")
    
    NGEN = 50
    POPSIZE = 136
    SEED = 5
    
    gde3 = GDE3(pop_size=POPSIZE, variant=f"DE/{selection}/1/{crossover}", CR=0.2, F=(0.0, 1.0), gamma=1e-4,
                survival=RankAndCrowding(crowding_func=crowding_func))

    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    assert len(res_gde3.opt) > 0
    

def test_many_perf():
    
    problem = get_problem("dtlz2")
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=15)
    igd = IGD(pf=problem.pareto_front(), zero_to_one=True)
    
    NGEN = 150
    POPSIZE = 136
    SEED = 5
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.2, F=(0.0, 1.0), gamma=1e-4,
                survival=RankAndCrowding(crowding_func="mnn"))

    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_gde3 = igd.do(res_gde3.F)
    assert igd_gde3 <= 0.07
    
    nsder = NSDER(ref_dirs=ref_dirs, pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 1.0), gamma=1e-4)
    
    res_nsder = minimize(problem,
                        nsder,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_nsder = igd.do(res_nsder.F)
    assert igd_nsder <= 0.01