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
from pymoo.util.vf_dominator import VFDominator
from pymoo.util.misc import has_feasible
from pymoo.util.reference_direction import select_points_with_maximum_distance
from pymoo.util import value_functions as mvf
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

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
                 output=MultiObjectiveOutput(),
                 tau=10,
                 eta=4,
                 **kwargs):
        
        self.survival = survival=RankAndCrowding(nds=NonDominatedSorting(dominator=VFDominator(self)))

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=self.survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

        self.tau = tau
        self.eta = eta
        self.eta_F = []
        self.vf_res = None
        self.v2 = None
        self.vf_plot_flag = False
        self.vf_plot = None
        self.historical_F = None

    @staticmethod
    def _prompt_for_ranks(F):

        for (e, f) in enumerate(F):
            print("Solution %d %s" % (e + 1, f * -1))   

        raw_ranks = input("Ranks (e.g., 3, 2, ..., 1): ")

        ranks = [int(raw_rank) for raw_rank in raw_ranks.split()  ] 

        return ranks

    @staticmethod
    def _get_ranks(F):

        ranks_invalid = True

        print("Rank the given solutions from highest to lowest preference:")

        ranks = PINSGA2._prompt_for_ranks(F)

        while ranks_invalid: 

            fc = F.shape[0]

            if sorted(ranks) == list(range(1,fc+1)):

                ranks_invalid = False 

            else: 

                print("Invalid ranks given. Please try again")

                ranks = PINSGA2._prompt_for_ranks(F)

        return ranks;                         
    


    def _advance(self, infills=None, **kwargs):

        super()._advance(infills=infills, **kwargs)

        F = self.pop.get("F")

        if self.historical_F is not None:
            self.historical_F = np.vstack((self.historical_F, F)) 
        else: 
            self.historical_F = F

        # Eta is the number of solutions displayed to the DM
        eta_F_indices = select_points_with_maximum_distance(self.pop.get("F"), self.eta)

        self.eta_F = F[eta_F_indices]
        self.eta_F = self.eta_F[self.eta_F[:,0].argsort()]

        # A frozen view of the optimization each 10 generations 
        self.paused_F = F

        if self.n_gen % 10 == 0:

            ranks = PINSGA2._get_ranks(self.eta_F)

            # ES or scimin
            approach = "ES"

            # linear or poly
            fnc_type = "poly"

            # max (False) or min (True)
            minimize = False

            if fnc_type == "linear":

                vf_res = mvf.create_linear_vf(self.eta_F * -1, ranks, approach, minimize)

            elif fnc_type == "poly":

                vf_res = mvf.create_poly_vf(self.eta_F * -1, ranks, approach, minimize)

            else:

                print("function not supported")

            self.vf_res = vf_res
            self.vf_plot_flag = True
            self.v2 = self.vf_res.vf(self.eta_F[ranks.index(2), :] * -1).item()
            

parse_doc_string(PINSGA2.__init__)


