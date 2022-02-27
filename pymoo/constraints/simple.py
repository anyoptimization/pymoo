import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.constraints.penalty import AdaptivePenalty


class SimpleAdaptivePenalty(AdaptivePenalty):

    def __init__(self,
                 algorithm,
                 nth_gen=5,
                 perc_pop=0.05,
                 beta_1=1.75,
                 beta_2=1.5,
                 **kwargs):
        """

        Parameters
        ----------
        algorithm

        nth_gen

        perc_pop

        beta_1 : float
            Increasing Rate
        beta_2 : float
            Decreasing Rate

        """

        assert beta_1 > 1 and beta_2 > 1, "Both rates need to be bigger than one."
        assert beta_1 > beta_2, "The authors recommended that the increasing is larger than the decreasing rate " \
                                "(also never equal to avoid cycles)"

        super().__init__(algorithm, **kwargs)

        self.nth_gen = nth_gen
        self.perc_pop = perc_pop

        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.feas_best = []

    def adapt_penalty(self, infills):
        n_best = int(np.ceil(self.perc_pop * len(infills)))

        best = FitnessSurvival().do(self.problem, infills, n_survive=n_best)

        self.feas_best.append(best.get("feas"))

        if len(self.feas_best) == self.nth_gen:

            alpha = self.penalty.alpha

            # if top solutions have non-zero penalty (were infeasible) -> increase penalty
            if np.all(~np.array(self.feas_best)):
                alpha *= self.beta_1

            # if top solutions have zero penalty (were feasible) -> decrease penalty
            elif np.all(np.array(self.feas_best)):
                alpha /= self.beta_2

            # make sure alpha is less or equal its maximum desired value
            alpha = min(alpha, self.max_alpha)

            self.penalty.alpha = alpha

            self.feas_best = []
