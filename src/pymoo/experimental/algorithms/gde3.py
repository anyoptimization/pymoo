from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.de import DE, Variant
from pymoo.core.population import Population

from pymoo.docs import parse_doc_string
from pymoo.operators.control import NoParameterControl
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import get_relation
from pymoo.termination.default import DefaultMultiObjectiveTermination


class GDE3(DE):

    def __init__(self, variant=None, **kwargs):

        if variant is None:

            if "control" not in kwargs:
                kwargs["control"] = NoParameterControl

            # the setting as proposed in the paper
            variant = Variant(selection="rand", F=0.1, CR=0.9, **kwargs)

        super().__init__(variant=variant, output=MultiObjectiveOutput(), **kwargs)
        self.termination = DefaultMultiObjectiveTermination()

    def _initialize_advance(self, infills=None, **kwargs):
        RankAndCrowdingSurvival().do(self.problem, infills, return_indices=True)

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus 'infills' must to be provided."
        pop = self.pop

        # the pool of solutions considered to survive
        pool = pop.tolist()

        # now do the replacement of individuals
        for infill in infills:
            k = infill.get("index")

            # get the relation between the infill and the solution from the population
            rel = get_relation(infill, pop[k])

            # if the new solution is not dominated by the individual
            if rel <= 0:
                pool.append(infill)

            # if the individual is not dominated by the new solution
            if rel >= 0:
                pool.append(pop[k])

        # set the rank and crowding in the current population
        pool = Population.create(*pool)
        self.pop = RankAndCrowdingSurvival().do(self.problem, pool, n_survive=self.pop_size)


parse_doc_string(GDE3.__init__)
