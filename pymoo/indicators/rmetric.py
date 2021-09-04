import numpy as np
from scipy.spatial.distance import cdist

from pymoo.core.indicator import Indicator
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD


class RMetric(Indicator):

    def __init__(self, problem, ref_points, w=None, delta=0.2, pf=None):
        """

        Parameters
        ----------
        
        problem : class
            problem instance
            
        ref_points : numpy.array
            list of reference points

        w : numpy.array
            weights for each objective

        delta : float
            The delta value representing the region of interest

        """

        Indicator.__init__(self)
        self.ref_points = ref_points
        self.problem = problem
        w_ = np.ones(self.ref_points.shape[1]) if not w else w
        self.w_points = self.ref_points + 2 * w_
        self.delta = delta

        self.pf = pf
        self.F = None
        self.others = None

    def _filter(self):

        def check_dominance(a, b, n_obj):
            flag1 = False
            flag2 = False
            for i in range(n_obj):
                if a[i] < b[i]:
                    flag1 = True
                else:
                    if a[i] > b[i]:
                        flag2 = True
            if flag1 and not flag2:
                return 1
            elif not flag1 and flag2:
                return -1
            else:
                return 0

        num_objs = np.size(self.F, axis=1)
        index_array = np.zeros(np.size(self.F, axis=0))

        # filter out all solutions that are dominated by solutions found by other algorithms
        if self.others is not None:
            for i in range(np.size(self.F, 0)):
                for j in range(np.size(self.others, 0)):
                    flag = check_dominance(self.F[i, :], self.others[j, :], num_objs)
                    if flag == -1:
                        index_array[i] = 1
                        break

        final_index = np.logical_not(index_array)
        filtered_pop = self.F[final_index, :]

        return filtered_pop

    def _preprocess(self, data, ref_point, w_point):

        datasize = np.size(data, 0)

        # Identify representative point
        ref_matrix = np.tile(ref_point, (datasize, 1))
        w_matrix = np.tile(w_point, (datasize, 1))
        # ratio of distance to the ref point over the distance between the w_point and the ref_point
        diff_matrix = (data - ref_matrix) / (w_matrix - ref_matrix)
        agg_value = np.amax(diff_matrix, axis=1)
        idx = np.argmin(agg_value)
        zp = [data[idx, :]]

        return zp,

    def _translate(self, zp, trimmed_data, ref_point, w_point):
        # Solution translation - Matlab reproduction
        # find k
        temp = ((zp[0] - ref_point) / (w_point - ref_point))
        kIdx = np.argmax(temp)

        # find zl
        temp = (zp[0][kIdx] - ref_point[kIdx]) / (w_point[kIdx] - ref_point[kIdx])
        zl = ref_point + temp * (w_point - ref_point)

        temp = zl - zp
        shift_direction = np.tile(temp, (trimmed_data.shape[0], 1))
        # new_size = self.curr_pop.shape[0]
        return trimmed_data + shift_direction

    def _trim(self, pop, centeroid, range=0.2):
        popsize, objDim = pop.shape
        diff_matrix = pop - np.tile(centeroid, (popsize, 1))[0]
        flags = np.sum(abs(diff_matrix) < range / 2, axis=1)
        filtered_matrix = pop[np.where(flags == objDim)]
        return filtered_matrix

    def _trim_fast(self, pop, centeroid, range=0.2):
        centeroid_matrix = cdist(pop, centeroid, metric='euclidean')
        filtered_matrix = pop[np.where(centeroid_matrix < range / 2), :][0]
        return filtered_matrix

    def do(self, F, others=None, calc_hv=True):
        """

        This method calculates the R-IGD and R-HV based off of the values provided.
        
        
        Parameters
        ----------

        F : numpy.ndarray
            The objective space values

        others : numpy.ndarray
            Results from other algorithms which should be used for filtering nds solutions

        calc_hv : bool
            Whether the hv is calculate - (None if more than 3 dimensions)


        Returns
        -------
        rigd : float
            R-IGD

        rhv : float
            R-HV if calc_hv is true and less or equal to 3 dimensions

        """
        self.F, self.others = F, others

        translated = []
        final_PF = []

        # 1. Prescreen Procedure - NDS Filtering
        pop = self._filter()

        pf = self.pf
        if pf is None:
            pf = self.problem.pareto_front()

        if pf is None:
            raise Exception("Please provide the Pareto front to calculate the R-Metric!")

        labels = np.argmin(cdist(pop, self.ref_points), axis=1)

        for i in range(len(self.ref_points)):
            cluster = pop[np.where(labels == i)]
            if len(cluster) != 0:
                # 2. Representative Point Identification
                zp = self._preprocess(cluster, self.ref_points[i], w_point=self.w_points[i])[0]
                # 3. Filtering Procedure - Filter points
                trimmed_data = self._trim(cluster, zp, range=self.delta)
                # 4. Solution Translation
                pop_t = self._translate(zp, trimmed_data, self.ref_points[i], w_point=self.w_points[i])
                translated.extend(pop_t)

            # 5. R-Metric Computation
            target = self._preprocess(data=pf, ref_point=self.ref_points[i], w_point=self.w_points[i])
            PF = self._trim(pf, target)
            final_PF.extend(PF)

        translated = np.array(translated)
        final_PF = np.array(final_PF)

        rigd, rhv = None, None

        if len(translated) > 0:

            # IGD Computation
            rigd = IGD(final_PF).do(translated)

            nadir_point = np.amax(self.w_points, axis=0)
            front = translated
            dim = self.ref_points[0].shape[0]
            if calc_hv:
                if dim <= 3:
                    try:
                        rhv = Hypervolume(ref_point=nadir_point).do(front)
                    except:
                        pass

        if calc_hv:
            return rigd, rhv
        else:
            return rigd
