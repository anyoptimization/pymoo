from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.docs import parse_doc_string
from pymoo.model.population import Population
from pymoo.model.survival import Survival
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import get_relation


class GDESurvival(Survival):

    def _do(self, problem, pop, n_survive, **kwargs):
        pass


class GDE3(DE):

    def __init__(self, **kwargs):
        super().__init__(display=MultiObjectiveDisplay(), **kwargs)

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus 'infills' must to be provided."

        survivors = []

        for k in range(self.pop_size):
            parent, off = self.pop[k], infills[k]

            rel = get_relation(parent, off)

            if rel == 0:
                survivors.extend([parent, off])
            elif rel == -1:
                survivors.append(off)
            else:
                survivors.append(parent)

        survivors = Population.create(*survivors)

        if len(survivors) > self.pop_size:
            survivors = RankAndCrowdingSurvival().do(self.problem, survivors, n_survive=self.pop_size)

        self.pop = survivors


parse_doc_string(GDE3.__init__)
