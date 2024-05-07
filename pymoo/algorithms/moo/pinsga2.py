import numpy as np
import warnings

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible

from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival


# =========================================================================================================
# Implementation
# =========================================================================================================


class PINSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowding(),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'
        #self.survival=RankAndCrowding()


    def _advance(self, infills=None, **kwargs):

        super()._advance(infills=infills, **kwargs)
      
        if self.n_gen % 5 == 0:

            print("What's your favorite color?")
            color = input()

            print("Neat! %s Is a great color." % color)


parse_doc_string(PINSGA2.__init__)
