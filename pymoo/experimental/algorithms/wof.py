from copy import deepcopy

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.docs import parse_doc_string
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.termination.default import MultiObjectiveDefaultTermination


# ---------------------------------------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------------------------------------


class WOF(Algorithm):

    def __init__(self,
                 gamma=4,
                 groups=2,
                 psi=3,
                 t1=1000,
                 t2=500,
                 q=None,
                 delta=0.5,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """
        WOF

        Parameters
        ----------


        """

        super().__init__(display=display, **kwargs)
        self.default_termination = MultiObjectiveDefaultTermination()

        self.proto = NSGA2()
        self.algorithm = None
        self.weighted = None
        self.n_cycle_evals = 0
        self.mode = "default"

        self.gamma = gamma
        self.groups = groups
        self.psi = psi
        self.t1 = t1
        self.t2 = t2
        self.q = q
        self.delta = delta

    def _setup(self, problem, **kwargs):
        self.algorithm = deepcopy(self.proto)
        self.algorithm.setup(problem)

        if self.q is None:
            self.q = problem.n_obj + 1

    def _initialize_infill(self):
        return self.algorithm.infill()

    def _initialize_advance(self, infills=None, **kwargs):
        self.algorithm.advance(infills=infills, **kwargs)
        self.n_cycle_evals = len(infills)

    def _infill(self):
        if self.mode == "weighted":
            return self._infill_weighted()
        else:
            return self._infill_default()

    def _infill_weighted(self):

        if self.weighted is None:
            self.weighted = deepcopy(self.proto)
            self.weighted.setup(self.proto)

            x_primes = self._x_primes(self.algorithm.pop, self.q)

            for x_prime in x_primes:

                groups = self._create_groups(x_prime)

        print("sdfsf")

    def _infill_default(self):
        return self.algorithm.infill()

    def _x_primes(self, pop, n):
        return Population.create(*np.random.choice(pop, n))

    def _create_groups(self, x_prime):
        pass

    def _advance(self, infills=None, **kwargs):

        # update the current number of evaluations in this cycle
        self.n_cycle_evals += len(infills)

        if self.mode == "weighted":
            # advance the algorithm which is currently used for mating
            self.weighted.advance(infills=infills, **kwargs)

            # remap the infills to be used for the default
            infills = self._map_to_default(infills)

            # check whether we shall switch back to the default cycle
            if self.n_cycle_evals > self.t2:
                self.weighted = None
                self.n_cycle_evals = 0
                self.mode = "default"


        elif self.mode == "default":

            # check if we switch to the weighted part of the algorithm
            if self.n_cycle_evals > self.t1:
                self.n_cycle_evals = 0
                self.mode = "weighted"

        # advance the inner algorithms to continue working
        self.algorithm.advance(infills=infills, **kwargs)


parse_doc_string(WOF.__init__)

if __name__ == "__main__":
    from pymoo.factory import get_problem
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter

    problem = get_problem("zdt1", n_var=100)

    algorithm = WOF()

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 200),
                   seed=1,
                   verbose=False)

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()
