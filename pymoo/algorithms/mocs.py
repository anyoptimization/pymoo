import math
import numpy as np
from itertools import combinations

from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.so_cuckoo_search import CuckooSearch
from pymoo.docs import parse_doc_string
from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination
from pymoo.model.initialization import Initialization
from pymoo.model.population import Population
from pymoo.model.replacement import ImprovementReplacement, is_better
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import has_feasible

class MOCSDisplay(MultiObjectiveDisplay):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)

class MOCS(CuckooSearch):

    def __init__(self,
                 display=MOCSDisplay(),
                 sampling=FloatRandomSampling(),
                 survival=RankAndCrowdingSurvival(),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 termination=None,
                 pop_size=100,
                 beta=1.5,
                 a0=0.1,
                 pa=0.35,
                 **kwargs):
        """

        Parameters
        ----------
        display : {display}
        sampling : {sampling}
        survival : {survival}
        eliminate_duplicates: {eliminate_duplicates}
        termination : {termination}

        pop_size : The number of nests (solutions)

        beta : The input parameter of the Mantegna's Algorithm to simulate
            sampling on Levy Distribution

        a0   : The scaling step size and is usually O(L/100) with L is the
            scale of the problem

        pa   : The switch probability, pa fraction of the nests will be
            abandoned on every iteration
        """

        super().__init__(display=display,
                         sampling=sampling,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         termination=termination,
                         pop_size=pop_size,
                         beta=beta,
                         a0=a0,
                         pa=pa,
                         **kwargs)

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]

    def _step(self):
        pop = self.pop
        X = pop.get("X")
        F = pop.get("F")

        #Levy Flight
        #pick the best one from random optimum nests (leas infeasibles or PF members)
        best = self.opt[np.random.randint(len(self.opt), size=len(X))]
        G_X = np.array([best_nest.get("X") for best_nest in best])

        step_size = self._get_global_step_size(X)
        _X = X + np.random.rand(*X.shape)*step_size*(G_X-X)
        _X = set_to_bounds_if_outside_by_problem(self.problem, _X)

        #Evaluate
        off = Population(len(_X)).set("X", _X)
        self.evaluator.eval(self.problem, off, algorithm=self)

        #Local Random Walk
        _X = off.get("X")
        step_size = self._get_local_step_size(X)
        _X = _X + step_size
        _X = set_to_bounds_if_outside_by_problem(self.problem, _X)
        off = Population(len(_X)).set("X", _X)
        self.evaluator.eval(self.problem, off, algorithm=self)

        #append offspring to population and then sort for elitism (survival)
        self.pop = Population.merge(pop, off)
        self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self)


parse_doc_string(MOCS.__init__)