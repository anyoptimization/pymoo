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
from pymoo.util.reference_direction import select_points_with_maximum_distance
from pymoo.util import value_functions as mvf
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival


# =========================================================================================================
# Helper functions
# =========================================================================================================

def get_relation(ind_a, ind_b):
    return Dominator.get_relation(ind_a.F, ind_b.F, ind_a.CV[0], ind_b.CV[0])

# =========================================================================================================
# Helper classes
# =========================================================================================================

class VFDominator:

    @staticmethod
    def get_relation(a, b, cva=None, cvb=None):

        if cva is not None and cvb is not None:
            if cva < cvb:
                return 1
            elif cvb < cva:
                return -1

        val = 0
        for i in range(len(a)):
            if a[i] < b[i]:
                # indifferent because once better and once worse
                if val == -1:
                    return 0
                val = 1
            elif b[i] < a[i]:
                # indifferent because once better and once worse
                if val == 1:
                    return 0
                val = -1
        return val

    @staticmethod
    def calc_domination_matrix_loop(F, G):
        n = F.shape[0]
        CV = np.sum(G * (G > 0).astype(float), axis=1)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = Dominator.get_relation(F[i, :], F[j, :], CV[i], CV[j])
                M[j, i] = -M[i, j]

        return M

    @staticmethod
    def calc_domination_matrix(F, _F=None, epsilon=0.0):

        print("Farts")
        if _F is None:
            _F = F

        # look at the obj for dom
        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

        M = np.logical_and(smaller, np.logical_not(larger)) * 1 \
            + np.logical_and(larger, np.logical_not(smaller)) * -1

        # if cv equal then look at dom
        # M = constr + (constr == 0) * dom

        return M

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
                 survival=RankAndCrowding(nds=NonDominatedSorting(dominator=VFDominator())),
                 output=MultiObjectiveOutput(),
                 tau=10,
                 eta=4,
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

        self.tau = tau
        self.eta = eta
        self.eta_F = []
        self.vf_res = None
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


parse_doc_string(PINSGA2.__init__)


