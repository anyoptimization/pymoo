import numpy as np

from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.model.algorithm import Algorithm, filter_optimum
from pymoo.model.individual import Individual
from pymoo.model.initialization import Initialization
from pymoo.model.population import Population
from pymoo.model.replacement import ImprovementReplacement
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.repair.out_of_bounds_repair import repair_out_of_bounds, repair_out_of_bounds_manually
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.misc import norm_eucl_dist, norm_euclidean_distance
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Implementation
# =========================================================================================================


class PSODisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        pop = algorithm.pop
        algorithm.pop = Population.create(*algorithm.pop.get("pbest"))
        super()._do(problem, evaluator, algorithm)
        algorithm.pop = pop

        if algorithm.adaptive:
            self.output.append("f", algorithm.f if algorithm.f is not None else "-", width=8)
            self.output.append("S", algorithm.strategy if algorithm.strategy is not None else "-", width=6)
            self.output.append("w", algorithm.w, width=6)
            self.output.append("c1", algorithm.c1, width=8)
            self.output.append("c2", algorithm.c2, width=8)


def S1_exploration(f):
    if f <= 0.4:
        return 0
    elif 0.4 < f <= 0.6:
        return 5 * f - 2
    elif 0.6 < f <= 0.7:
        return 1
    elif 0.7 < f <= 0.8:
        return -10 * f + 8
    elif 0.8 < f:
        return 0


def S2_exploitation(f):
    if f <= 0.2:
        return 0
    elif 0.2 < f <= 0.3:
        return 10 * f - 2
    elif 0.3 < f <= 0.4:
        return 1
    elif 0.4 < f <= 0.6:
        return -5 * f + 3
    elif 0.6 < f:
        return 0


def S3_convergence(f):
    if f <= 0.1:
        return 1
    elif 0.1 < f <= 0.3:
        return -5 * f + 1.5
    elif 0.3 < f:
        return 0


def S4_jumping_out(f):
    if f <= 0.7:
        return 0
    elif 0.7 < f <= 0.9:
        return 5 * f - 3.5
    elif 0.9 < f:
        return 1


class PSO(Algorithm):

    def __init__(self,
                 pop_size=20,
                 w=0.9,
                 c1=2.0,
                 c2=2.0,
                 sampling=LatinHypercubeSampling(),
                 adaptive=True,
                 pertube_best=True,
                 display=PSODisplay(),
                 repair=None,
                 individual=Individual(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}

        """

        super().__init__(display=display, **kwargs)

        self.initialization = Initialization(sampling,
                                             individual=individual,
                                             repair=repair)

        self.pop_size = pop_size
        self.adaptive = adaptive
        self.pertube_best = pertube_best
        self.default_termination = SingleObjectiveDefaultTermination()
        self.V_max = None

        self.w = w
        self.c1 = c1
        self.c2 = c2

    def initialize(self, problem, **kwargs):
        super().initialize(problem, **kwargs)
        self.V_max = 0.2 * (problem.xu - problem.xl)

    def _initialize(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        self.evaluator.eval(self.problem, pop, algorithm=self)

        if self.pertube_best:
            pop = FitnessSurvival().do(self.problem, pop, self.pop_size-1)

        pop.set("V", np.zeros((len(pop), self.problem.n_var)))
        pop.set("pbest", pop)
        self.pop = pop

        self.f = None
        self.strategy = None

    def _next(self):
        self._step()

        if self.adaptive:
            self._adapt()

    def _step(self):
        pop = self.pop
        X, F, V = pop.get("X", "F", "V")

        # get the personal best of each particle
        pbest = Population.create(*pop.get("pbest"))
        P_X, P_F = pbest.get("X", "F")

        # get the global best solution
        best = self.opt.repeat(len(pop))
        G_X = best.get("X")

        # perform the pso equation
        inerta = self.w * V

        # calculate random values for the updates
        r1 = np.random.random((len(pop), self.problem.n_var))
        r2 = np.random.random((len(pop), self.problem.n_var))

        cognitive = self.c1 * r1 * (P_X - X)
        social = self.c2 * r2 * (G_X - X)

        # calculate the velocity vector
        _V = inerta + cognitive + social
        _V = repair_out_of_bounds_manually(_V, - self.V_max, self.V_max)

        # update the values of each particle
        _X = X + _V
        _X = repair_out_of_bounds(self.problem, _X)

        # evaluate the offspring population
        off = Population(len(pop)).set("X", _X, "V", _V, "pbest", pbest)
        self.evaluator.eval(self.problem, off, algorithm=self)

        # check whether a solution has improved or not - also consider constraints here
        has_improved = ImprovementReplacement().do(self.problem, pbest, off, return_indices=True)

        # replace the personal best of each particle if it has improved
        off[has_improved].set("pbest", off[has_improved])
        off.set("best", best)
        pop = off

        # try to improve the current best with a pertubation
        if self.pertube_best:
            opt = FitnessSurvival().do(self.problem,  Population.create(*pop.get("pbest")), 1)
            eta = int(np.random.uniform(5, 30))
            mutant = PolynomialMutation(eta).do(self.problem, opt)
            self.evaluator.eval(self.problem, mutant, algorithm=self)
            if ImprovementReplacement().do(self.problem, opt, mutant, return_indices=True)[0]:
                k = [i for i, e in enumerate(pop.get("pbest")) if e == opt][0]
                pop[k].set("pbest", mutant)

        self.pop = pop

    def _adapt(self):
        pop = self.pop

        X, F, best = pop.get("X", "F", "best")
        best = Population.create(*best)
        w, c1, c2, = self.w, self.c1, self.c2

        # get the average distance from one to another for normalization
        D = norm_eucl_dist(self.problem, X, X)
        mD = D.sum(axis=1) / (len(pop) - 1)
        _min, _max = mD.min(), mD.max()

        # get the average distance to the global best
        g_D = norm_euclidean_distance(self.problem)(best.get("X"), X).mean()
        f = (g_D - _min) / (_max - _min + 1e-32)

        S = np.array([S1_exploration(f), S2_exploitation(f), S3_convergence(f), S4_jumping_out(f)])
        strategy = S.argmax() + 1

        delta = 0.05 + (np.random.random() * 0.05)

        if strategy == 1:
            c1 += delta
            c2 -= delta
        elif strategy == 2:
            c1 += 0.5 * delta
            c2 -= 0.5 * delta
        elif strategy == 3:
            c1 += 0.5 * delta
            c2 += 0.5 * delta
        elif strategy == 4:
            c1 -= delta
            c2 += delta

        c1 = max(1.5, min(2.5, c1))
        c2 = max(1.5, min(2.5, c2))

        if c1 + c2 > 4.0:
            c1 = 4.0 * (c1 / (c1 + c2))
            c2 = 4.0 * (c2 / (c1 + c2))

        w = 1 / (1 + 1.5 * np.exp(-2.6 * f))

        self.f = f
        self.strategy = strategy
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def _set_optimum(self, force=False):
        pbest = Population.create(*self.pop.get("pbest"))
        self.opt = filter_optimum(pbest, least_infeasible=True)


parse_doc_string(PSO.__init__)
