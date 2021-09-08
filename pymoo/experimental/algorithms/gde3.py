from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.population import Population
from pymoo.docs import parse_doc_string
from pymoo.operators.mutation.pm import PM
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import get_relation
from pymoo.util.termination.default import MultiObjectiveDefaultTermination


class GDE3(DE):

    def __init__(self,
                 CR=0.3,
                 F=None,
                 variant="DE/rand/1/bin",
                 mutation=PM(eta=15),
                 **kwargs):

        super().__init__(CR=CR, F=F, variant=variant, mutation=mutation, display=MultiObjectiveDisplay(), **kwargs)
        self.default_termination = MultiObjectiveDefaultTermination()

    def _initialize_advance(self, infills=None, **kwargs):
        pass

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus 'infills' must to be provided."

        # the indices for the creating of offsprings considered in the last generation
        I = infills.get("index")

        # the individuals that are considered for the survival later and final survive
        survivors = []

        # now for each of the infill solutions
        for k, i in enumerate(I):

            # get the offspring an the parent it is coming from
            off, parent = infills[k], self.pop[i]

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
        self.pop[I] = RankAndCrowdingSurvival().do(self.problem, survivors, n_survive=len(I))


parse_doc_string(GDE3.__init__)
