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
                 ranking_type='pairwise',
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
        
        self.ranking_type = ranking_type

        self.tau = tau
        self.eta = eta
        self.eta_F = []
        self.vf_res = None
        self.v2 = None
        self.vf_plot_flag = False
        self.vf_plot = None
        self.historical_F = None
        self.prev_pop = None

    @staticmethod
    def _prompt_for_ranks(F):

        for (e, f) in enumerate(F):
            print("Solution %d %s" % (e + 1, f))   

        raw_ranks = input("Ranks (e.g., 3, 2, ..., 1): ")

        ranks = [int(raw_rank) for raw_rank in raw_ranks.split()  ] 

        return ranks
        


    @staticmethod
    def _get_pairwise_ranks(F):

        # initialize empty ranking
        ranks = []
        offset = 0
        for i, f in enumerate( F ):
            
            # handle empty case, put first element in first place
            if not ranks:
                ranks.append( [i] )
                
            else:
                inserted = False
                
                # for each remaining elements, compare to all currently ranked elements
                for j, group in enumerate( ranks ):

                    # get pairwise preference from user
                    while True:
                        preference = input( f"\nWhich solution do you like best?\n[a] {f}\n[b] {F[ group[0] ]}\n[c] These solutions are equivalent.\n--> " ).strip().lower()
                        if preference in ['a', 'b', 'c']:
                            break
                        print("Invalid input. Please enter 'a', 'b', or 'c'.")
                    
                    # if better than currenly ranked element place before that element
                    if preference == 'a':
                        ranks.insert( j, [ i - offset ] )
                        inserted = True
                        break
                    
                    # if equal to currently ranked element place with that element
                    elif preference == 'c':
                        group.append( i - 1 - offset )
                        offset += 1
                        inserted = True
                        break
                    
                # if found to be worse than all place at the end
                if not inserted:
                    ranks.append( [ i - offset ] )
        
        _ranks = [ rank for group in ranks for rank in group ]
        
        # reorder ranks
        ranks = np.zeros ( len( _ranks ), dtype=int ) 

        for index, rank in enumerate( _ranks) :
            ranks[rank] = index
        
        return np.array( ranks )


    @staticmethod
    def _get_ranks(F):

        ranks_invalid = True

        print("Rank the given solutions from highest to lowest preference:")

        ranks = PINSGA2._prompt_for_ranks( F)

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

            if self.ranking_type == "absolute": 
                dm_ranks = PINSGA2._get_ranks(self.eta_F)
            elif self.ranking_type == "pairwise": 
                dm_ranks = PINSGA2._get_pairwise_ranks(self.eta_F)
                print(dm_ranks)
            else: 
                raise ValueError("Invalid ranking type [%s] given." % self.ranking_type)

            

            if len(set(rank)) == 0: 

                print("No preference between any two points provided.")

                self._reset_dm_preference()

                return 

            eta_F = self.eta_F

            while eta_F.shape[0] > 1:

                # ES or scimin
                approach = "ES"

                # linear or poly
                fnc_type = "poly"

                # max (False) or min (True)
                minimize = False

                if fnc_type == "linear":

                    vf_res = mvf.create_linear_vf(eta_F * -1, dm_ranks.tolist(), approach, minimize)

                elif fnc_type == "poly":

                    vf_res = mvf.create_poly_vf(eta_F * -1, dm_ranks.tolist(), approach, minimize)

                else:

                    print("function not supported")

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


