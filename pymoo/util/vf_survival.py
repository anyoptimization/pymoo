import numpy as np

from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

class VFSurvival( RankAndCrowding ):
    
    def __init__( self, dominator ):
        
        
        nds = VFNonDominatedSorting( dominator )
        
        super().__init__( nds, crowding_func="cd" )
           
class VFNonDominatedSorting():
    
    def __init__( self, dominator, epsilon=None ) -> None:
        super().__init__()
        self.dominator = dominator
        self.epsilon = epsilon

    def do(self, F, return_rank=False, only_non_dominated_front=False, n_stop_if_ranked=None, **kwargs):
        F = F.astype(float)

        # if not set just set it to a very large values because the cython algorithms do not take None
        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)

        # set the epsilon if it should be set
        if self.epsilon is not None:
            kwargs["epsilon"] = float(self.epsilon)

        fronts = vf_non_dominated_sort( F, self.dominator )

        # convert to numpy array for each front and filter by n_stop_if_ranked if desired
        _fronts = []
        n_ranked = 0
        for front in fronts:

            _fronts.append(np.array(front, dtype=int))

            # increment the n_ranked solution counter
            n_ranked += len(front)

            # stop if more than this solutions are n_ranked
            if n_ranked >= n_stop_if_ranked:
                break

        fronts = _fronts

        if only_non_dominated_front:
            return fronts[0]

        if return_rank:
            rank = rank_from_fronts(fronts, F.shape[0])
            return fronts, rank

        return fronts


def rank_from_fronts(fronts, n):
    # create the rank array and set values
    rank = np.full(n, 1e16, dtype=int)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank
    
def vf_non_dominated_sort(F, dominator, **kwargs):

    M = dominator.calc_domination_matrix(F)


    # calculate the dominance matrix
    n = M.shape[0]

    fronts = []

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=int)

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n)

    current_front = []

    for i in range(n):

        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts

class VFDominator:

    def __init__(self, algorithm):

        self.algorithm = algorithm

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
                M[i, j] = VFDominator.get_relation(F[i, :], F[j, :], CV[i], CV[j])
                M[j, i] = -M[i, j]

        return M

    def calc_domination_matrix(self, F, _F=None, epsilon=0.0):

        if _F is None:
            _F = F

        # look at the obj for dom
        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))
    
        non_dom = np.logical_and(smaller, np.logical_not(larger))
        dom = np.logical_and(larger, np.logical_not(smaller))

        if self.algorithm.vf_res is not None: 

            # Figure out what the v2 value is 
            v2 = self.algorithm.v2

            # Get the value function
            vf = self.algorithm.vf_res.vf

            # How much does the DM value each solution?
            F_vf = vf(F * -1)[:,np.newaxis]
            _F_vf = vf(_F * -1)[:,np.newaxis]

            # We want to compare each solution to the others 
            Lv = np.repeat(F_vf, m, axis=0)
            Rv = np.tile(_F_vf, (n, 1))
            
            # Which values are greater than (better) V2? 
            gtv2 = np.reshape(Lv < v2, (n, m))
            # Which values are less than (worst) V2? 
            ltv2 = np.reshape(Rv > v2, (n, m))
         
            # If you are greater than V2, you dominate all those who are smaller than V2
            split_by_v2 = np.logical_and(gtv2, ltv2)
           
            dom = np.logical_or(dom, split_by_v2)

        M = non_dom * 1 \
            + dom * -1
       
        return M
 