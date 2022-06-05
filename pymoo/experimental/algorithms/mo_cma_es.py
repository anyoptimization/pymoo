import numpy as np

from deap.cma import StrategyMultiObjective
from deap import creator, base

from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
from pymoo.util.display import MultiObjectiveDisplay  # ,Display
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import pop_from_array_or_individual
from pymoo.core.population import Population

class MO_CMAES(StrategyMultiObjective, Algorithm):

    def __init__(self,
                 x0=None,
                 mu=None,
                 lambda_=1,
                 spring=1,
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
        self.spring = spring   
        self.x0 = x0

    def _setup(self, problem, x0=None, **kwargs):
        # print("_setup")
        if self.x0 is None:
            self.x0 = x0
        creator.create("FitnessMin",
                       base.Fitness,
                       weights=(-1.0, ) * problem.n_obj)
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.ind_init = creator.Individual
        
    def _initialize_infill(self):
        # print("_initialize_infill")
        # calculate the default number of sample points
        # no initial point is provided - sample in bounds and take the best
        if self.x0 is None:
            if not self.problem.has_bounds():
                raise Exception("Either provide an x0 or a problem with variable bounds!")
            pop = self.sampling.do(self.problem, self.mu)
        else:
            pop = pop_from_array_or_individual(self.x0)
        return pop

    def _infill(self):
        # print("_infill")
        self.pop_tmp = []
        for _ in range(self.spring):
            self.pop_tmp.extend( self.generate(self.ind_init) )

        pop = Population.new(
            "X",
            set_to_bounds_if_outside(np.asarray(self.pop_tmp), 
                                     self.problem.xl,
                                     self.problem.xu))
        return pop

    def _initialize_advance(self, infills=None, **kwargs):        
        # print("_initialize_advance")
        # we should have a populated and evalauted self.pop at this point
        # We can now instantiate our StrategyMultiObjective        
        population = [self.ind_init(f) for f in self.pop.get("X")]
        for ind, fit in zip(population, self.pop.get("F")):
            ind.fitness.values = fit
        StrategyMultiObjective.__init__(self,
                                        population,
                                        self.sigma,
                                        mu=self.mu,
                                        lambda_=self.lambda_,
                                        **self.params)

    def _advance(self, infills=None, **kwargs):        
        # print("_advance")   
        
        for ind in infills:
            if not ind.feasible[0]:
                ind.F[0] = np.nan
        inds = self.generate(self.ind_init)
        for ind, fit, values in zip(inds, 
                                    infills.get("F"),
                                    infills.get("X")):
            ind.clear()
            ind.extend(values)
            ind.fitness.values = fit
        self.update(inds)
        self.pop = Population.merge(self.pop, infills)