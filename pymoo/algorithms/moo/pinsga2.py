import sys
from abc import ABC, abstractmethod

import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util import value_functions as mvf
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.reference_direction import select_points_with_maximum_distance
from pymoo.util.vf_dominator import VFDominator


# =========================================================================================================
# Implementation
# =========================================================================================================


class AutomatedDM(ABC): 
    
    def __init__(self, get_pairwise_ranks_func=None):
        self.get_pairwise_ranks_func = get_pairwise_ranks_func

    @abstractmethod
    def makeDecision(self, F):
        pass
    
    def makePairwiseDecision(self, F):
        
        dm = lambda F: self.makeDecision(F)
        ranks = self.get_pairwise_ranks_func(F, 1, dm=dm)
        
        return ranks
        


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
                 eps_max=1000,
                 ranking_type='pairwise',
                 presi_signs=None,
                 automated_dm=None,
                 verbose=False, 
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
        
        self.ranking_type=ranking_type
        self.presi_signs=presi_signs

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
        self.eps_max = eps_max

        self.verbose = verbose

        if automated_dm is not None:
            automated_dm.get_pairwise_ranks_func = self._get_pairwise_ranks
        self.automated_dm = automated_dm

    def _warn(self, msg):
        if self.verbose: 
            sys.stderr.write(msg)

    @staticmethod
    def _prompt_for_ranks(F, presi_signs):

        for (e, f) in enumerate(F):
            print("Solution %d %s" % (e + 1, f * presi_signs))   

        dim = F.shape[0]                                                                                 

        raw_ranks = input(f"Ranks (e.g., \"3, {dim}, ..., 1\" for 3rd best, {dim}th best, ..., 1st best): ")

        if raw_ranks == "":
            ranks = []
        else:
            ranks = [int(raw_rank) for raw_rank in raw_ranks.split()  ] 

        return ranks
       
    @staticmethod
    def _present_ranks(F, dm_ranks, presi_signs):

        print("Solutions are ranked as:")

        for (e, f) in enumerate(F):
            print("Solution %d %s: Rank %d" % (e + 1, f * presi_signs, dm_ranks[e]))   


    @staticmethod
    def _get_pairwise_ranks(F, presi_signs, dm=None):

        if not dm:
                                  
            dm = lambda F: input("\nWhich solution do you like best?\n" + \
                                    f"[a] {F[0]}\n" +  \
                                    f"[b] {F[1]}\n" + \
                                     "[c] These solutions are equivalent.\n--> " )

        # initialize empty ranking
        _ranks = []
        for i, f in enumerate( F ):
            
            # handle empty case, put first element in first place
            if not _ranks:
                _ranks.append( [i] )
                
            else:
                inserted = False
                
                # for each remaining elements, compare to all currently ranked elements
                for j, group in enumerate( _ranks ):

                    # get pairwise preference from user
                    while True:

                        points_to_compare = np.array( [f*presi_signs, F[ group[0] ]*presi_signs] )
                        preference_raw = dm( points_to_compare )

                        preference = preference_raw.strip().lower()

                        if preference in ['a', 'b', 'c']:
                            break
                        print("Invalid input. Please enter 'a', 'b', or 'c'.")
                    
                    # if better than currently ranked element place before that element
                    if preference == 'a':
                        _ranks.insert( j, [i] )
                        inserted = True
                        break
                    
                    # if equal to currently ranked element place with that element
                    elif preference == 'c':
                        group.append( i )
                        inserted = True
                        break
                    
                # if found to be worse than all place at the end
                if not inserted:
                    _ranks.append( [i] )

        ranks = np.zeros ( len( F ), dtype=int ) 

        for rank, group in enumerate( _ranks ):
            for index in group:
                ranks[index] = rank

        return np.array( ranks ) + 1


    @staticmethod
    def _get_ranks(F, presi_signs):

        ranks_invalid = True

        dim = F.shape[0]                                                                                 
                                                                                                          
        print(f"Give each solution a ranking, with 1 being the highest score, and {dim} being the lowest score:")        

        ranks = PINSGA2._prompt_for_ranks(F, presi_signs)

        while ranks_invalid: 

            fc = F.shape[0]

            if len(ranks) > 0 and max(ranks) <= fc and min(ranks) >= 1:

                ranks_invalid = False 

            else: 

                print("Invalid ranks given. Please try again")

                ranks = PINSGA2._prompt_for_ranks(F, presi_signs)

        return np.array(ranks);                         


    def _reset_dm_preference(self):

            self._warn("Back-tracking and removing DM preference from search.")

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

        if self.presi_signs is None: 
            self.presi_signs = np.ones(F.shape[1])

        # Eta is the number of solutions displayed to the DM
        eta_F_indices = select_points_with_maximum_distance(F, to_find, random_state=self.random_state)

        self.eta_F = F[eta_F_indices]
        self.eta_F = self.eta_F[self.eta_F[:,0].argsort()]

        # Remove duplicate rows
        self.eta_F = np.unique(self.eta_F, axis=0)

        # A frozen view of the optimization each tau generations 
        self.paused_F = F

        # Record the previous population in case we need to back track 
        self.prev_pop = self.pop

        dm_time = self.n_gen % self.tau == 0

        # Check whether we have more than one solution
        if dm_time and len(self.eta_F) < 2: 

            self._warn("Population only contains one non-dominated solution. ")

            self._reset_dm_preference()

        elif dm_time:
       
            # Check if the DM is a machine or a human
            if self.automated_dm is None: 

                # Human DM
                if self.ranking_type == "absolute": 
                    dm_ranks = PINSGA2._get_ranks(self.eta_F, self.presi_signs)
                elif self.ranking_type == "pairwise": 
                    dm_ranks = PINSGA2._get_pairwise_ranks(self.eta_F, self.presi_signs)
                    PINSGA2._present_ranks(self.eta_F, dm_ranks, self.presi_signs) 
                else: 
                    raise ValueError("Invalid ranking type [%s] given." % self.ranking_type)
            else:

                # Automated DM
                if self.ranking_type == "absolute": 
                    dm_ranks = self.automated_dm.makeDecision(self.eta_F)
                elif self.ranking_type == "pairwise": 
                    dm_ranks = self.automated_dm.makePairwiseDecision(self.eta_F)
                else: 
                    raise ValueError("Invalid ranking type [%s] given." % self.ranking_type)

            

            if len(set(rank)) == 0: 

                self._warn("No preference between any two points provided.")

                self._reset_dm_preference()

                return 

            eta_F = self.eta_F

            while eta_F.shape[0] > 1:

                if self.vf_type == "linear":

                    vf_res = mvf.create_linear_vf(eta_F * -1, 
                                                  dm_ranks.tolist(), 
                                                  eps_max=self.eps_max, 
                                                  method=self.opt_method)

                elif self.vf_type == "poly":

                    vf_res = mvf.create_poly_vf(eta_F * -1, 
                                                dm_ranks.tolist(), 
                                                eps_max=self.eps_max, 
                                                method=self.opt_method)
    
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
                    self._warn("Could not fit a function to the DM preference")

                    if eta_F.shape[0] == 2:

                        # If not, reset and use normal domination
                        self._warn("Removing DM preference")
                        self._reset_dm_preference()
                        break

                    else:

                        self._warn("Removing the second best preferred solution from the fit.")

                        # ranks start at 1, not zero
                        rank_to_remove = dm_ranks[1] 
                        eta_F = np.delete(eta_F, rank_to_remove - 1, axis=0)

                        dm_ranks = np.concatenate(([dm_ranks[0]], dm_ranks[2:]))
                        
                        # update the ranks, since we just removed one
                        dm_ranks[dm_ranks > rank_to_remove] = dm_ranks[dm_ranks > rank_to_remove] - 1 


parse_doc_string(PINSGA2.__init__)


