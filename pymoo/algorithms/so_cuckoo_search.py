import math
import numpy as np
from itertools import combinations

from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination
from pymoo.model.initialization import Initialization
from pymoo.model.population import Population
from pymoo.model.replacement import ImprovementReplacement, is_better
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

class CSDisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)

class CuckooSearch(Algorithm):

    def __init__(self,
                 display=CSDisplay(),
                 sampling=FloatRandomSampling(),
                 survival=FitnessSurvival(),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 termination=SingleObjectiveDefaultTermination(),
                 pop_size=100,
                 beta=1.5,
                 alfa=0.01,
                 pa=0.35,
                 **kwargs):
        """

        Parameters
        ----------
        display : {display}
        sampling : {sampling}
        survival : {survival}
        eliminate_duplicates: This does not exists in the original paper/book.
            Without this the solutions might get too biased to current global best solution,
            because the global random walk use the global best solution as the reference.

        termination : {termination}

        pop_size : The number of nests (solutions)

        beta : The input parameter of the Mantegna's Algorithm to simulate
            sampling on Levy Distribution

        alfa : alfa is the step size scaling factor and is usually
            0.01, so that the step size will be scaled down to O(L/100) with L is
            the scale (range of bounds) of the problem.

        pa   : The switch probability, pa fraction of the nests will be
            abandoned on every iteration
        """

        super().__init__(**kwargs)

        self.initialization = Initialization(sampling)
        self.survival = survival
        self.display = display
        self.pop_size = pop_size
        self.default_termination = termination
        self.eliminate_duplicates = eliminate_duplicates

        #the scale will be multiplied by problem scale after problem given in setup
        self.alfa = alfa
        self.scale = alfa
        self.pa = pa
        self.beta = beta
        a = math.gamma(1. + beta) * math.sin(math.pi*beta/2.)
        b = beta*math.gamma((1.+beta)/2.)*2**((beta-1.)/2)
        self.sig = (a/b)**(1./(2*beta))



    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)
        x_lower, x_upper = self.problem.bounds()
        if x_lower is not None and x_upper is not None:
            self.scale = self.alfa*(x_upper-x_lower)
        else:
            self.scale = self.alfa

    def _initialize(self):
        pop = self.initialization.do(self.problem,
                                     self.pop_size,
                                     algorithm=self,
                                     eliminate_duplicates=self.eliminate_duplicates)
        self.evaluator.eval(self.problem, pop, algorithm=self)

        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)
        self.pop = pop

    def _next(self):
        self._step()

    def _get_levy_step(self, shape):
        #Mantegna's algorithm simulating levy sampling
        U = np.random.normal(0, self.sig, shape)
        V = abs(np.random.normal(0, 1, shape))**(1./self.beta)
        return U/V

    def _get_global_step_size(self, X):
        step = self._get_levy_step(X.shape)
        step_size = self.scale*step
        return step_size

    def _get_local_directional_vector(self, X):
        #local random walk (abandon nest) for pa fraction of the nests
        #find 2 random different solution for the local random walk (nest_i ~ nest_i+ (nest_j - nest_k))
        Xjk_idx = np.random.rand(len(X), len(X)).argpartition(2, axis=1)[:, :2]
        Xj_idx = Xjk_idx[:, 0]
        Xk_idx = Xjk_idx[:, 1]
        Xj = X[Xj_idx]
        Xk = X[Xk_idx]

        #calculate Heaviside function (or wether local search will be done with nest_i or not)
        #then duplicate H coloumn as many as the number of decision variable
        H = (np.random.rand(len(X))<self.pa).astype(np.float)
        H = np.tile(H, (self.problem.n_var, 1)).transpose()

        #calculate d (scale*(X_j - X_k)) , however XS Yang implementation in mathworks differ from the book
        #replacing the scale with a random number [0,1], we use the book version here (a0)
        dir_vec = np.random.rand(*X.shape)*(Xj-Xk)*H
        return dir_vec


    def _step(self):
        pop = self.pop
        X = pop.get("X")
        F = pop.get("F")

        #Levy Flight
        best = self.opt
        G_X = best.get("X")

        step_size = self._get_global_step_size(X)
        _X = X + np.random.rand(*X.shape)*step_size*(G_X-X)
        _X = set_to_bounds_if_outside_by_problem(self.problem, _X)

        #Evaluate
        off = Population(len(pop)).set("X", _X)
        self.evaluator.eval(self.problem, off, algorithm=self)

        # replace the worse pop with better off per index
        # this method includes replacement with less constraints violation
        # which the original paper doesn't have
        ImprovementReplacement().do(self.problem, pop, off, inplace=True)

        #Local Random Walk
        dir_vec = self._get_local_directional_vector(X)
        _X = X + dir_vec
        _X = set_to_bounds_if_outside_by_problem(self.problem, _X)
        off = Population(len(pop)).set("X", _X)
        self.evaluator.eval(self.problem, off, algorithm=self)

        #append offspring to population and then sort for elitism (survival)
        self.pop = Population.merge(pop, off)
        self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self)

parse_doc_string(CuckooSearch.__init__)
