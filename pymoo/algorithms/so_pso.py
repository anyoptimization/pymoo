import numpy as np

from pymoo.docs import parse_doc_string
from pymoo.model.algorithm import Algorithm, filter_optimum
from pymoo.model.individual import Individual
from pymoo.model.initialization import Initialization
from pymoo.model.population import Population
from pymoo.operators.repair.out_of_bounds_repair import repair_out_of_bounds, repair_out_of_bounds_manually
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.util.display import Display
from pymoo.util.misc import norm_eucl_dist
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Implementation
# =========================================================================================================


class PSODisplay(Display):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("fopt", algorithm.opt[0].F[0])

        if algorithm.adaptive:
            self.output.append("f", algorithm.f if algorithm.f is not None else "-", width=8)
            self.output.append("S", algorithm.strategy if algorithm.strategy is not None else "-", width=6)
            self.output.append("w", algorithm.w, width=6)
            self.output.append("c1", algorithm.c1, width=8)
            self.output.append("c2", algorithm.c2, width=8)


def S1_exploration(f):
    if 0.0 <= f <= 0.4:
        return 0
    elif 0.4 < f <= 0.6:
        return 5 * f - 2
    elif 0.6 < f <= 0.7:
        return 1
    elif 0.7 < f <= 0.8:
        return -10 * f + 8
    elif 0.8 < f <= 1.0:
        return 0


def S2_exploitation(f):
    if 0.0 <= f <= 0.2:
        return 0
    elif 0.2 < f <= 0.3:
        return 10 * f - 2
    elif 0.3 < f <= 0.4:
        return 1
    elif 0.4 < f <= 0.6:
        return -5 * f + 3
    elif 0.6 < f <= 1.0:
        return 0


def S3_convergence(f):
    if 0.0 <= f <= 0.1:
        return 1
    elif 0.1 < f <= 0.3:
        return -5 * f + 1.5
    elif 0.3 < f <= 1.0:
        return 0


def S4_jumping_out(f):
    if 0.0 <= f <= 0.7:
        return 0
    elif 0.7 < f <= 0.9:
        return 5 * f - 3.5
    elif 0.9 < f <= 1.0:
        return 1


class PSO(Algorithm):

    def __init__(self,
                 pop_size=20,
                 w=0.9,
                 c1=2.0,
                 c2=2.0,
                 sampling=LatinHypercubeSampling(),
                 adaptive=True,
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
        G_I = P_F.argmin()
        G_X = P_X[G_I]

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

        # replace the personal best of each particle if it has improved
        has_improved = (off.get("F") < P_F)[:, 0]
        off[has_improved].set("pbest", off[has_improved])
        pop = off

        self.pop = pop

    def _adapt(self):
        # this might be different from the paper - pbest instead of the particle directly
        # pop = Population.create(*self.pop.get("pbest"))
        pop = self.pop

        X, F = pop.get("X", "F")
        w, c1, c2, = self.w, self.c1, self.c2

        D = norm_eucl_dist(self.problem, X, X)
        mD = D.sum(axis=1) / (len(pop) - 1)

        _min, _max = mD.min(), mD.max()
        g = F[:, 0].argmin()
        f = (mD[g] - _min) / (_max - _min)

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
