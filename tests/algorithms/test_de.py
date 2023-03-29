import pytest

import numpy as np

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.moo.gde3 import GDE3
from pymoo.algorithms.moo.nsde import NSDE
from pymoo.algorithms.moo.nsder import NSDER
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.dex import DEX
from pymoo.operators.crossover.dem import DEM
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding


@pytest.mark.parametrize('selection', ["rand", "best", "current-to-best", "current-to-rand", "ranked"])
@pytest.mark.parametrize('crossover', ["bin", "exp"])
@pytest.mark.parametrize('de_repair', ["bounce-back", "midway", "rand-init", "to-bounds"])
def test_de_run(selection, crossover, de_repair):
    problem = get_problem("rastrigin")
    
    NGEN = 20
    POPSIZE = 20
    SEED = 3
    
    #DE Parameters
    CR = 0.5
    F = (0.3, 1.0)

    de = DE(pop_size=POPSIZE, variant=f"DE/{selection}/1/{crossover}", CR=CR, F=F, de_repair=de_repair)

    res_de = minimize(problem,
                    de,
                    ('n_gen', NGEN),
                    seed=SEED,
                    save_history=False,
                    verbose=False)

    assert len(res_de.opt) > 0


def test_de_pm_run():
    
    problem = get_problem("rastrigin")
    
    NGEN = 20
    POPSIZE = 20
    SEED = 3
    
    #DE Parameters
    CR = 0.5
    F = (0.3, 1.0)

    de = DE(pop_size=POPSIZE, variant=f"DE/rand/1/bin", CR=CR, F=F)

    res_de = minimize(problem,
                      de,
                      ('n_gen', NGEN),
                      seed=SEED,
                      save_history=False,
                      verbose=False)
    
    depm = DE(pop_size=POPSIZE, variant=f"DE/rand/1/bin", CR=CR, F=F, genetic_mutation=PM())

    res_pm = minimize(problem,
                      depm,
                      ('n_gen', NGEN),
                      seed=SEED,
                      save_history=False,
                      verbose=False)

    assert len(res_pm.opt) > 0
    assert len(sum(res_de.pop.get("F") - res_pm.pop.get("F"))) >= 1e-6
    

def test_de_perf():
    problem = get_problem("rastrigin")
    
    NGEN = 100
    POPSIZE = 20
    SEED = 3
    
    #DE Parameters
    CR = 0.5
    F = (0.3, 1.0)

    de = DE(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=CR, F=F)

    res_de = minimize(problem,
                    de,
                    ('n_gen', NGEN),
                    seed=SEED,
                    save_history=False,
                    verbose=False)

    assert len(res_de.opt) > 0
    assert res_de.F <= 1e-6


@pytest.mark.parametrize('survival', [RankAndCrowding])
@pytest.mark.parametrize('crowding_func', ["mnn", "2nn", "cd", "pcd", "ce"])
def test_gde3_run_metric(survival, crowding_func):
    
    problem = get_problem("truss2d")
    
    gde3 = GDE3(pop_size=50, survival=survival(crowding_func=crowding_func))

    res = minimize(
        problem,
        gde3,
        ('n_gen', 30),
        seed=12,
        save_history=False,
        verbose=False,
    )
    
    assert len(res.opt) > 0


@pytest.mark.parametrize('selection', ["rand", "best", "current-to-best", "current-to-rand", "ranked"])
@pytest.mark.parametrize('crossover', ["bin", "exp"])
def test_gde3_run_variant(selection, crossover):
    
    problem = get_problem("truss2d")
    
    gde3 = GDE3(pop_size=50, variant=f"DE/{selection}/1/{crossover}")

    res = minimize(
        problem,
        gde3,
        ('n_gen', 30),
        seed=12,
        save_history=False,
        verbose=False,
    )
    
    assert len(res.opt) > 0


def test_nsde_run():
    
    problem = get_problem("truss2d")
    
    algorithm = NSDE(pop_size=100)

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 20),
        seed=12,
        save_history=False,
        verbose=False,
    )
    
    assert len(res.opt) > 0


def test_nsder_run():
    
    problem = get_problem("dtlz2")

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=15)
    algorithm = NSDER(ref_dirs=ref_dirs, pop_size=136)

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 20),
        seed=12,
        verbose=False,
    )
    
    assert len(res.opt) > 0


@pytest.mark.parametrize('crossover', [DEX(), DEM()])
def test_multi_frankstein(crossover):
    
    problem = get_problem("truss2d")

    NGEN = 50
    POPSIZE = 5
    SEED = 5
    
    frank = NSGA2(pop_size=POPSIZE, crossover=crossover)

    res_frank = minimize(problem,
                        frank,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    assert len(res_frank.opt) > 0


def test_gde3_pm_run():
    
    problem = get_problem("truss2d")

    NGEN = 30
    POPSIZE = 50
    SEED = 5
    
    gde3pm = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back",
                survival=RankAndCrowding(crowding_func="pcd"), genetic_mutation=PM())

    res_gde3pm = minimize(problem,
                          gde3pm,
                          ('n_gen', NGEN),
                          seed=SEED,
                          save_history=False,
                          verbose=False)
    
    assert len(res_gde3pm.opt) > 0
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back",
                survival=RankAndCrowding(crowding_func="pcd"))
    
    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    assert len(sum(res_gde3pm.F - res_gde3.F)) >= 1e-3


@pytest.mark.parametrize('selection', ["rand", "current-to-rand", "ranked"])
@pytest.mark.parametrize('crossover', ["bin", "exp"])
@pytest.mark.parametrize('crowding_func', ["mnn", "2nn"])
def test_many_run(selection, crossover, crowding_func):
    
    problem = get_problem("dtlz2")
    
    NGEN = 10
    POPSIZE = 100
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
