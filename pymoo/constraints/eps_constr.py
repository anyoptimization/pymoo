import numpy as np

from pymoo.algorithms.base.meta import MetaAlgorithm
from pymoo.constraints.tcv import estm_scale


class AdaptiveConstraintHandling(MetaAlgorithm):

    def __init__(self, algorithm, perc_eps_until=0.75, **kwargs):
        super().__init__(algorithm, **kwargs)
        self.perc_eps_until = perc_eps_until
        self.max_cv = None

    def _setup(self, *args, **kwargs):
        super()._setup(*args, **kwargs)

        # make sure the tcv is attached to each individual and changes dynamically
        self.evaluator.attach_tcv = True

    def _initialize_advance(self, infills=None, **kwargs):
        G, H = infills.get("G", "H")

        # add the normalization constants based on the first set of infills
        if G.shape[1] > 0:
            self.tcv.ieq_scale = np.array([estm_scale(g) for g in G.T])
        if H.shape[1] > 0:
            self.tcv.eq_scale = np.array([estm_scale(h) for h in H.T])

        # get the average constraint violation in the current generation
        CV = self.pop.get("CV")
        self.max_cv = np.mean(CV)

        # initialize the allowed cv for counting as feasible
        self.tcv.feas_eps = self.max_cv

        super()._initialize_advance(infills=infills, **kwargs)

    def _advance(self, infills=None, **kwargs):
        t = self.n_gen / self.termination.n_max_gen

        alpha = np.maximum(0.0, 1 - 1 / self.perc_eps_until * t)
        self.tcv.feas_eps = alpha * self.max_cv

        super()._advance(infills, **kwargs)


