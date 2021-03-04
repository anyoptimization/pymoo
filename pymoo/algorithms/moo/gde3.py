from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.docs import parse_doc_string
from pymoo.model.population import Population
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import get_relation
from pymoo.util.termination.default import MultiObjectiveDefaultTermination


class GDE3(DE):

    def __init__(self, CR=0.9, F=0.1, variant="DE/rand/1/bin", **kwargs):
        super().__init__(CR=CR, F=F, variant=variant, display=MultiObjectiveDisplay(), **kwargs)
        self.default_termination = MultiObjectiveDefaultTermination()

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus 'infills' must to be provided."

        # the individuals that are considered for the survival later and finall survive
        survivors = []

        # the indices for the creating of offsprings considered in the last generation
        H = {}
        for i, j in enumerate(self.indices):
            H[j] = i

        for i in range(len(self.pop)):
            parent = self.pop[i]

            # this is only relevant if n_offsprings is set and only a few solutions should be replaced
            if i not in H:
                survivors.append(parent)

            else:

                off = infills[H[i]]

                # check whether the new solution dominates the parent or not
                rel = get_relation(parent, off)

                # if indifferent we add both
                if rel == 0:
                    survivors.extend([parent, off])

                # if offspring dominates parent
                elif rel == -1:
                    survivors.append(off)

                # if parent dominates offspring
                else:
                    survivors.append(parent)

        # create the population
        survivors = Population.create(*survivors)

        # perform a survival to reduce to pop size
        if len(survivors) > self.pop_size:
            survivors = RankAndCrowdingSurvival().do(self.problem, survivors, n_survive=self.pop_size)

        self.pop = survivors


parse_doc_string(GDE3.__init__)
