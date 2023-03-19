import pytest

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.igd import IGD
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
def test_multi_run(survival, crowding_func):
    
    problem = get_problem("truss2d")

    NGEN = 30
    POPSIZE = 50
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
    assert abs(igd_gde3 - 0.005859828655308572) <= 1e-8
    
    gde3p = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back",
                survival=RankAndCrowding(crowding_func="pcd"))

    res_gde3p = minimize(problem,
                        gde3p,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_gde3p = igd.do(res_gde3p.F)
    assert abs(igd_gde3p - 0.004744463013355145) <= 1e-8
    
    nsde = NSDE(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back",
                survival=RankAndCrowding(crowding_func="pcd"))
        
    res_nsde = minimize(problem,
                        nsde,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_nsde = igd.do(res_nsde.F)
    assert abs(igd_nsde - 0.004562068055351625) <= 1e-8


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
    

def test_many_perf():
    
    np.random.seed(3)
    assert abs(np.random.rand() - 0.5507979025745755) <= 1e-8
    
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
    assert abs(igd_gde3 - 0.04011488503871424) <= 1e-8
    
    nsder = NSDER(ref_dirs=ref_dirs, pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 1.0), gamma=1e-4)
    
    res_nsder = minimize(problem,
                        nsder,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_nsder = igd.do(res_nsder.F)
    assert abs(igd_nsder - 0.004877000918527632) <= 1e-8