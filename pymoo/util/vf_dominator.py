import numpy as np

from pymoo.util.dominator import Dominator


def get_relation(ind_a, ind_b):
    return Dominator.get_relation(ind_a.F, ind_b.F, ind_a.CV[0], ind_b.CV[0])


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
                M[i, j] = Dominator.get_relation(F[i, :], F[j, :], CV[i], CV[j])
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



