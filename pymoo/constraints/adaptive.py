import numpy as np

from pymoo.algorithms.base.meta import MetaAlgorithm
from pymoo.constraints.tcv import TotalConstraintViolation


class AdaptiveConstraintHandling(MetaAlgorithm):

    def __init__(self, algorithm, perc_eps_until=0.75, **kwargs):
        super().__init__(algorithm, **kwargs)
        self.perc_eps_until = perc_eps_until

    def _setup(self, *args, **kwargs):
        super()._setup(*args, **kwargs)

        # make sure the tcv is attached to each individual and changes dynamically
        self.evaluator.attach_tcv = True

        # the allowed constraint values for each ieq and eq constraints
        self.ieq_eps, self.eq_eps = None, None

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)

        G, H = infills.get("G", "H")
        ieq_scale = np.mean(np.maximum(0.0, G), axis=0)
        eq_scale = np.mean(np.maximum(0.0, H), axis=0)

        self.tcv = TotalConstraintViolation(ieq_eps=ieq_scale,
                                            ieq_pow=None,
                                            ieq_scale=ieq_scale,
                                            eq_eps=eq_scale,
                                            eq_pow=None,
                                            eq_scale=eq_scale,
                                            aggr_func=np.mean)

        self.ieq_eps, self.eq_eps = ieq_scale, eq_scale

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills, **kwargs)

        t = self.n_gen / self.termination.n_max_gen

        alpha = np.maximum(0.0, 1 - 1 / self.perc_eps_until * t)
        self.tcv.ieq_eps = alpha * self.ieq_eps

        eq_eps = 1e-5
        beta = np.maximum(eq_eps, 1 - 1 / (self.perc_eps_until+eq_eps) * t)
        self.tcv.eq_eps = beta * self.eq_eps
