import numpy as np

from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.core.population import Population, calc_cv
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.functions import load_function


class StochasticRankingSurvival(Survival):

    def __init__(self, PR):
        super().__init__(filter_infeasible=False)
        self.PR = PR

    def _do(self, problem, pop, *args, n_survive=None, tcv=None, random_state=None, **kwargs):
        assert problem.n_obj == 1, "This stochastic ranking implementation only works for single-objective problems."

        F, G = pop.get("F", "G")
        f = F[:, 0]

        if not problem.has_constraints():
            I = f.argsort()

        else:
            phi = calc_cv(pop)
            J = np.arange(len(phi))
            I = load_function("stochastic_ranking")(f, phi, self.PR, J, random_state=random_state)

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

    def _advance(self, infills=None, **kwargs):

        # if not all solutions suggested by infill() are evaluated we create a more semi (mu+lambda) algorithm
        if len(infills) < self.pop_size:
            infills = Population.merge(infills, self.pop)

        self.pop = self.survival.do(self.problem, infills, n_survive=self.pop_size)


parse_doc_string(SRES.__init__)
