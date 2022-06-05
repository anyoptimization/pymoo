from cvxpy import Problem
import numpy as np

from deap.cma import StrategyMultiObjective
from deap import creator, base

from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
from pymoo.util.display import MultiObjectiveDisplay  # ,Display
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import pop_from_array_or_individual
from pymoo.core.population import Population

class MO_CMAES(Algorithm):

    def __init__(self,                 
                 mu,
                 lambda_=1,                 
                 sigma=0.1,
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

    def _setup(self, problem, **kwargs):
        creator.create("FitnessMin",
                       base.Fitness,
                       weights=(-1.0, ) * problem.n_obj)
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.ind_init = creator.Individual

    def _initialize_infill(self):
        return self.sampling.do(Problem, self.mu)

    def _initialize_advance(self, infills=None, **kwargs):        
        population = [self.ind_init(f) for f in infills.get("X")]
        #Must have set n_initial_doe to mu in surrogate
        assert len(population) == self.mu
        for ind, fit in zip(population, self.pop.get("F")):
            ind.fitness.values = fit
        self.mo_cma_es = StrategyMultiObjective(population,
                                                self.sigma,
                                                mu=self.mu,
                                                lambda_=self.lambda_,
                                                **self.params)
        self.pop = infills

    def _infill(self):       
        population = self.mo_cma_es.generate(self.ind_init)
        infills = Population.new("X",
                    set_to_bounds_if_outside(np.asarray(population),
                                            self.problem.xl,
                                            self.problem.xu))
        for inf, pop in zip(infills, population):
            inf.org_pop = pop
        return infills

    def _advance(self, infills=None, **kwargs):
        for ind in infills:
            if not ind.feasible[0]:
                ind.F[0] = np.nan
        population = [inf.org_pop for inf in infills]
        # assert len(self.mo_cma_es.generate(self.ind_init)) == infills.shape[0]
        for ind, fit, values in zip(population,
                                    infills.get("F"),
                                    infills.get("X")):
            # ind.clear()
            # ind.extend(values)
            ind.fitness.values = fit
        self.mo_cma_es.update(population)
        self.pop = infills