from cvxpy import Problem
import numpy as np

from deap.cma import StrategyMultiObjective
from deap import creator, base

from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
from pymoo.util.display import MultiObjectiveDisplay  # ,Display
from pymoo.util.optimum import filter_optimum
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import pop_from_array_or_individual
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival

class MO_CMAES(StrategyMultiObjective, Algorithm):

    def __init__(self,            
                 mu,
                 momo=False,
                 lambda_=1,                 
                 sigma=0.1,
                 mu0=None,
                 sampling=LatinHypercubeSampling(),
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        Algorithm.__init__(self, display=display, **kwargs)
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma
        self.params = kwargs
        self.sampling= sampling       
        self.mo_cma_es = None 
        if mu0 is None:
            self.mu0 = mu
        else:
            self.mu0 = mu0
            assert self.mu <= self.mu0
        self.survival = RankAndCrowdingSurvival()

    def _setup(self, problem, **kwargs):
        creator.create("FitnessMin",
                       base.Fitness,
                       weights=(-1.0, ) * problem.n_obj)
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.ind_init = creator.Individual

    def _initialize_infill(self):
        return self.sampling.do(self.problem, self.mu0)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        assert len(infills) == self.mu0
        infills = self.survival.do(self.problem, infills, n_survive=self.mu)
        population = [self.ind_init(f) for f in infills.get("X")]
        #Must have set n_initial_doe to mu0 in surrogate
        assert len(population) == self.mu
        for ind, fit in zip(population, self.pop.get("F")):
            ind.fitness.values = fit
            # print(ind, ind.fitness)
        StrategyMultiObjective.__init__(self,
                                        population,
                                        self.sigma,
                                        mu=self.mu,
                                        lambda_=self.lambda_,
                                        **self.params)       
        self.pop = infills 

    def _infill(self):       
        population = self.generate(self.ind_init)
        infills = Population.new("X",
                            set_to_bounds_if_outside(np.asarray(population),
                                                     self.problem.xl,
                                                     self.problem.xu))
        for inf, pop in zip(infills, population):
            inf.org_pop = pop
        return infills

    def _advance(self, infills=None, **kwargs):
        for inf in infills:
            if not inf.feasible[0]:
                inf.F[0] = np.nan
                print(inf, "Solution not feasible")
        population = [inf.org_pop for inf in infills if inf]
        # try:
        #     assert len(population) == self.lambda_
        # except:
        #     print(len(population), self.lambda_)
        #     raise
        self.generate(self.ind_init)
        for ind, fit, values in zip(population,
                                    infills.get("F"),
                                    infills.get("X")):
            ind.clear()
            ind.extend(values)
            ind.fitness.values = fit
        # for ind in population:
        #     print("-",ind, ind._ps, ind.fitness)
        self.update(population)
        self.pop = infills

    def _set_optimum(self):
        sols = self.pop
        if self.opt is not None:
            sols = Population.merge(sols, self.opt)
        self.opt = filter_optimum(sols, least_infeasible=True)
