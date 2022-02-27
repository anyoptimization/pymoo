import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.core.replacement import is_better
from pymoo.core.variable import Real, Integer, get, Choice
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.pcx import PCX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.rnd import fast_fill_random
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Implementation
# =========================================================================================================


class G3PCX(Algorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 n_offsprings=2,
                 n_parents=3,
                 family_size=2,
                 repair=NoRepair(),
                 display=SingleObjectiveDisplay(),
                 **kwargs):

        super().__init__(display=display, **kwargs)

        self.pop_size = Integer(pop_size, bounds=(20, 200))
        self.repair = repair

        self.default_termination = SingleObjectiveDefaultTermination()

        self.initialization = Initialization(sampling, repair=self.repair, eliminate_duplicates=False)

        self.n_offsprings = Integer(n_offsprings, bounds=(1, 10))
        self.n_parents = Integer(n_parents, bounds=(3, 10))
        self.family_size = Integer(family_size, bounds=(1, 10))

        self.crossover = PCX()
        self.crossover.prob = 1.0

        self.mutation = PM()
        self.mutation.prob = Real(0.25, bounds=(0.0, 1.0))

        self.cnt = 1

    def _initialize_infill(self):
        return self.initialization.do(self.problem, get(self.pop_size), algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = FitnessSurvival().do(self.problem, infills, n_survive=len(infills), algorithm=self, **kwargs)

    def _infill(self):
        n_offsprings, n_parents = get(self.n_offsprings, self.n_parents)

        S = np.zeros((n_offsprings, n_parents), dtype=int)
        S[:, 0] = 0
        fast_fill_random(S, len(self.pop), columns=range(1, n_parents))

        off = self.crossover.do(self.problem, self.pop, parents=S, algorithm=self)

        off = self.mutation.do(self.problem, off, algorithm=self)

        self.repair.do(self.problem, off, algorithm=self)

        return off

    def _advance(self, infills=None, **kwargs):
        pop, family_size = self.pop, get(self.family_size)

        rnd = np.random.choice(np.arange(len(pop)), size=family_size, replace=False)
        family = Population.merge(pop[rnd], infills)
        pop[rnd] = FitnessSurvival().do(self.problem, family, n_survive=family_size)

        for i in rnd:
            if is_better(pop[i], pop[0]):
                tmp = pop[0]
                pop[0] = pop[i]
                pop[i] = tmp

        # only print the output every 100 evaluations (otherwise that is far too much)
        mod_n_eval = 100

        self.display.active = False
        cnt = self.evaluator.n_eval // mod_n_eval
        if cnt > self.cnt:
            self.display.active = True
            self.cnt = cnt

        #
        self.n_gen = (self.evaluator.n_eval // get(self.pop_size)) - 1

    def _set_optimum(self, **kwargs):
        self.opt = self.pop[[0]]


parse_doc_string(G3PCX.__init__)
