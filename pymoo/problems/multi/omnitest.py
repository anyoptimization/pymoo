import numpy as np
from numpy import sum, pi, sin, cos

from pymoo.model.problem import Problem


class OmniTest(Problem):
    """
    The Omni-test problem proposed by Deb in [1].

    Parameters
    ----------
    n_var: number of decision variables

    References
    ----------
    [1] Deb, K., Tiwari, S. "Omni-optimizer: A generic evolutionary algorithm for single and multi-objective optimization"
    """
    def __init__(self, n_var=2):
        assert (n_var >= 2), "The dimension of the decision space should at least be 2!"
        super().__init__(
            n_var=n_var, n_obj=2, n_constr=0, type_var=np.double,
            xl=np.full(n_var, 0), xu=np.full(n_var, 6)
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F1 = sum(sin(pi * X), axis=1)
        F2 = sum(cos(pi * X), axis=1)
        out["F"] = np.vstack((F1, F2)).T

    def _calc_pareto_set(self, n_pareto_points=500):
        # The Omni-test problem has 3^D Pareto subsets
        num_ps = int(3 ** self.n_var)
        h = int(n_pareto_points / num_ps)
        PS = np.zeros((num_ps * h, self.n_var))

        candidates = np.array([np.linspace(2 * m + 1, 2 * m + 3 / 2, h) for m in range(3)])
        # generate combination indices
        candidates_indices = [[0, 1, 2] for _ in range(self.n_var)]
        a = np.meshgrid(*candidates_indices)
        combination_indices = np.array(a).T.reshape(-1, self.n_var)
        # generate 3^D combinations
        for i in range(num_ps):
            PS[i * h:i * h + h, :] = candidates[combination_indices[i]].T
        return PS

    def _calc_pareto_front(self, n_pareto_points=500):
        PS = self._calc_pareto_set(n_pareto_points)
        return self.evaluate(PS, return_values_of=["F"])
