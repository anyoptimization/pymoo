from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.so_de import DE
from pymoo.docs import parse_doc_string
from pymoo.model.population import Population
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import get_relation


class GDE3(DE):

    def __init__(self, **kwargs):
        super().__init__(display=MultiObjectiveDisplay(), **kwargs)

    def _next(self):

        # make a step and create the offsprings
        self.off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        self.off.set("n_gen", self.n_gen)

        # evaluate the offsprings
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        survivors = []

        for k in range(self.pop_size):
            parent, off = self.pop[k], self.off[k]

            rel = get_relation(parent, off)

            if rel == 0:
                survivors.extend([parent, off])
            elif rel == -1:
                survivors.append(off)
            else:
                survivors.append(parent)

        survivors = Population.create(*survivors)

        if len(survivors) > self.pop_size:
            survivors = RankAndCrowdingSurvival().do(self.problem, survivors, self.pop_size)

        self.pop = survivors


parse_doc_string(GDE3.__init__)
