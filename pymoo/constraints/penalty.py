import numpy as np

from pymoo.algorithms.base.meta import MetaAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.constraints.tcv import TotalConstraintViolation
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.problems.meta import MetaProblem
from pymoo.util.display import Display
from pymoo.util.misc import at_least_2d_array
from pymoo.util.optimum import filter_optimum


# =========================================================================================================
# STATIC - Redefine the problem with a static penalty
# =========================================================================================================


class ConstraintsAsPenalty(MetaProblem):

    def __init__(self,
                 problem,
                 tcv,
                 penalty: float = 0.1,
                 f_ideal=None,
                 f_scale=None
                 ):

        super().__init__(problem)

        # the amount of penalty to add for this type
        self.penalty = penalty

        # the cv calculator to obtain the constraint violation from G and H
        self.tcv = tcv

        # normalization parameters for the objective value(s)
        self.f_ideal = f_ideal
        self.f_scale = f_scale

        # set ieq and eq to zero (because it became now a penalty)
        self.n_ieq_constr = 0
        self.n_eq_constr = 0

    def do(self, x, out, *args, **kwargs):
        self.problem.do(x, out, *args, **kwargs)

        if self.problem.has_constraints():

            # get at the values from the output
            F, G, H = at_least_2d_array(out["F"]), at_least_2d_array(out["G"]), at_least_2d_array(out["H"])

            # store a backup of the values in out
            out["__F__"], out["__G__"], out["__H__"] = F, G, H

            # normalize the objective space values if more information is provided
            if self.f_ideal is not None:
                F = F - self.f_ideal
            if self.f_scale is not None:
                F = F / self.f_scale

            # calculate the total constraint violation (here normalization shall be already included)
            cv = self.tcv.calc(G, H)

            # set the penalized objective values
            out["F"] = F + self.penalty * cv[:, None]

            # erase the values from the output - it is unconstrained now
            out.pop("G", None), out.pop("H", None)


# =========================================================================================================
# Adaptive
# =========================================================================================================


class Penalty:

    def __init__(self, alpha, tcv) -> None:
        """
        An object consisting of the total constraint violation and penalty to calculated the new obj. values

        Parameters
        ----------
        alpha : float
            Penalty when applying f + alpha * cv

        tcv : class
            Calculate to determine the total constraint violation.
        """
        super().__init__()
        self.alpha = alpha
        self.tcv = tcv
        self.active = True


class PenalizedIndividual(Individual):

    def __init__(self, penalty) -> None:
        """
        An individual that changes its function values depending on the settings of the penalty object.
        Parameters
        ----------
        penalty : class

        """
        self.penalty = penalty

    @property
    def F(self):
        if self.penalty.active:
            tcv = self.penalty.tcv.calc(self.G, self.H)
            return self._F + self.penalty.alpha * tcv
        else:
            return self._F

    @F.setter
    def F(self, value):
        self._F = value

    @property
    def CV(self):
        if self.penalty.active:
            return np.array([0.0])
        else:
            return self._CV

    @CV.setter
    def CV(self, value):
        self._CV = value

    def new(self):
        return self.__class__(self.penalty)


# =========================================================================================================
# Adaptive Penalty - Change the penalty during the algorithm execution
# =========================================================================================================


class AdaptivePenaltyDisplay(Display):

    def __init__(self, display, **kwargs):
        super().__init__(**kwargs)
        self.display = display

    def _do(self, problem, evaluator, algorithm):
        self.display.do(problem, evaluator, algorithm, show=False)
        self.output = self.display.output
        self.output.append("penalty", algorithm.penalty.alpha)


class CleanAdaptivePenalty(MetaAlgorithm):

    def __init__(self, algorithm, alpha=2.0, max_alpha=1e6, **kwargs):
        super().__init__(algorithm, **kwargs)
        tcv = TotalConstraintViolation()
        self.penalty = Penalty(alpha, tcv)
        self.max_alpha = max_alpha

    def _setup(self, problem, **kwargs):
        self.evaluator.individual = PenalizedIndividual(self.penalty)
        self.display = AdaptivePenaltyDisplay(self.display)

    def _initialize_infill(self):
        self.penalty.active = True
        infills = super()._initialize_infill()
        self.penalty.active = False
        return infills

    def _initialize_advance(self, infills=None, **kwargs):
        self.initial_penalty(infills)

        self.penalty.active = True
        super()._initialize_advance(infills, **kwargs)
        self.penalty.active = False

    def _infill(self):
        self.penalty.active = True
        ret = super()._infill()
        self.penalty.active = False
        return ret

    def _advance(self, infills=None, **kwargs):
        self.adapt_penalty(infills)

        self.penalty.active = True
        super()._advance(infills, **kwargs)
        self.penalty.active = False

    def _set_optimum(self):
        pop = self.pop
        if self.opt is not None:
            pop = Population.merge(pop, self.opt)
        self.opt = filter_optimum(pop, least_infeasible=True)

    def _finalize(self):
        self.penalty.active = True
        super()._finalize()
        self.penalty.active = False

    def initial_penalty(self, infills):
        pass

    def adapt_penalty(self, infills):
        pass


class AdaptivePenalty(MetaAlgorithm):

    def __init__(self, algorithm, alpha=2.0, max_alpha=1e6,
                 nth_gen=5,
                 perc_pop=0.000001,
                 beta_1=1.5,
                 beta_2=1.3,
                 **kwargs):

        super().__init__(algorithm, **kwargs)

        tcv = TotalConstraintViolation()
        self.penalty = Penalty(alpha, tcv)
        self.max_alpha = max_alpha

    def _setup(self, problem, **kwargs):
        self.evaluator.individual = PenalizedIndividual(self.penalty)
        self.display = AdaptivePenaltyDisplay(self.display)

    def _initialize_infill(self):
        self.penalty.active = True
        infills = super()._initialize_infill()
        self.penalty.active = False
        return infills

    def _initialize_advance(self, infills=None, **kwargs):
        self.initial_penalty(infills)

        self.penalty.active = True
        super()._initialize_advance(infills, **kwargs)
        self.penalty.active = False

    def _infill(self):
        self.penalty.active = True
        ret = super()._infill()
        self.penalty.active = False
        return ret

    def _advance(self, infills=None, **kwargs):
        self.adapt_penalty(infills)

        self.penalty.active = True
        super()._advance(infills, **kwargs)
        self.penalty.active = False

    def _set_optimum(self):
        pop = self.pop
        if self.opt is not None:
            pop = Population.merge(pop, self.opt)
        self.opt = filter_optimum(pop, least_infeasible=True)

    def _finalize(self):
        self.penalty.active = True
        super()._finalize()
        self.penalty.active = False

    def initial_penalty(self, infills):
        pass

    def adapt_penalty(self, infills):
        pass