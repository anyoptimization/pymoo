import numpy as np
import warnings

from abc import ABC, abstractmethod
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


class AutomatedDM(ABC): 

    @abstractmethod
    def makeDecision(self, F):
        pass


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
                 opt_method="trust-constr",
                 vf_type="poly",
                 automated_dm=None,
                 **kwargs):
        
        self.survival = RankAndCrowding(nds=NonDominatedSorting(dominator=VFDominator(self)))

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

        self.vf_type = vf_type
        self.opt_method = opt_method
        self.tau = tau
        self.eta = eta
        self.eta_F = []
        self.vf_res = None
        self.v2 = None
        self.vf_plot_flag = False
        self.vf_plot = None
        self.historical_F = None
        self.prev_pop = None
        self.fronts = []

        self.automated_dm=automated_dm

    @staticmethod
    def _prompt_for_ranks(F):

        for (e, f) in enumerate(F):
            print("Solution %d %s" % (e + 1, f))   

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

        return np.array(ranks);                         
    
    def _reset_dm_preference(self):

            print("Back-tracking and removing DM preference from search.")

            self.eta_F = []
            self.vf_res = None
            self.v2 = None
            self.vf_plot_flag = False
            self.vf_plot = None
            self.pop = self.prev_pop


    def _advance(self, infills=None, **kwargs):

        super()._advance(infills=infills, **kwargs)

        rank, F = self.pop.get("rank", "F")

        self.fronts = rank

        F = F[rank == 0]

        if self.historical_F is not None:
            self.historical_F = np.vstack((self.historical_F, F)) 
        else: 
            self.historical_F = F

        to_find = self.eta if F.shape[0] >= self.eta else F.shape[0] 

        # Eta is the number of solutions displayed to the DM
        eta_F_indices = select_points_with_maximum_distance(F, to_find)

        self.eta_F = F[eta_F_indices]
        self.eta_F = self.eta_F[self.eta_F[:,0].argsort()]

        # Remove duplicate rows
        self.eta_F = np.unique(self.eta_F, axis=0)

        # A frozen view of the optimization each 10 generations 
        self.paused_F = F

        # Record the previous population in case we need to back track 
        self.prev_pop = self.pop

        dm_time = self.n_gen % 10 == 0

        # Check whether we have more than one solution
        if dm_time and len(self.eta_F) < 2: 

            print("Population only contains one non-dominated solution. ")

            self._reset_dm_preference()

        elif dm_time:
        
            if self.automated_dm == None: 
                dm_ranks = PINSGA2._get_ranks(self.eta_F)
            else:
                dm_ranks = self.automated_dm.makeDecision(self.eta_F)

            

            if len(set(rank)) == 0: 

                print("No preference between any two points provided.")

                self._reset_dm_preference()

                return 

            eta_F = self.eta_F

            while eta_F.shape[0] > 1:

                if self.vf_type == "linear":

                    vf_res = mvf.create_linear_vf(eta_F * -1, dm_ranks.tolist(), self.opt_method)

                elif self.vf_type == "poly":

                    vf_res = mvf.create_poly_vf(eta_F * -1, dm_ranks.tolist(), self.opt_method)

                else:
                    
                    raise ValueError("Value function %s not supported" % self.vf_type)

                # check if we were able to model the VF
                if vf_res.fit:

                    self.vf_res = vf_res
                    self.vf_plot_flag = True
                    self.v2 = self.vf_res.vf(eta_F[dm_ranks[1] - 1] * -1).item()
                    break

                else:

                    # If we didn't the model, try to remove the least preferred point and try to refit
                    print("Could not fit a function to the DM preference")

                    if eta_F.shape[0] == 2:

                        # If not, reset and use normal domination
                        print("Removing DM preference")
                        self._reset_dm_preference()

                    else:

                        print("Removing the second best preferred solution from the fit.")

                        # ranks start at 1, not zero
                        rank_to_remove = dm_ranks[1] 
                        eta_F = np.delete(eta_F, rank_to_remove - 1, axis=0)

                        dm_ranks = np.concatenate(([dm_ranks[0]], dm_ranks[2:]))
                        
                        # update the ranks, since we just removed one
                        dm_ranks[dm_ranks > rank_to_remove] = dm_ranks[dm_ranks > rank_to_remove] - 1 








parse_doc_string(PINSGA2.__init__)


