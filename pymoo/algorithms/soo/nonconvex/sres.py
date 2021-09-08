import numpy as np

from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.docs import parse_doc_string
from pymoo.core.survival import Survival
from pymoo.util.function_loader import load_function


class StochasticRankingSurvival(Survival):

    def __init__(self, PR):
        super().__init__(filter_infeasible=False)
        self.PR = PR

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        assert problem.n_obj == 1, "This stochastic ranking implementation only works for single-objective problems."

        F, G = pop.get("F", "G")
        f = F[:, 0]

        if problem.n_constr == 0:
            I = f.argsort()

        else:
            phi = (np.maximum(0, G) ** 2).sum(axis=1)
            J = np.arange(len(phi))
            I = load_function("stochastic_ranking")(f, phi, self.PR, J)

        return pop[I][:n_survive]


class SRES(ES):

    def __init__(self, PF=0.45, **kwargs):
        """
        Stochastic Ranking Evolutionary Strategy (SRES)

        Parameters
        ----------
        PF: float
            The stochastic ranking weight for choosing a random decision while doing the modified bubble sort.
        """
        super().__init__(survival=StochasticRankingSurvival(PF), **kwargs)
        self.PF = PF


parse_doc_string(SRES.__init__)
