from moocore import epsilon_additive as _epsilon_additive
from moocore import epsilon_mult as _epsilon_mult

from pymoo.core.indicator import Indicator
from pymoo.util.misc import at_least_2d_array
from pymoo.indicators.distance_indicator import derive_ideal_and_nadir_from_pf


class EpsilonIndicator(Indicator):

    def __init__(self, pf, **kwargs):
        pf = at_least_2d_array(pf, extend_as="row")
        ideal, nadir = derive_ideal_and_nadir_from_pf(pf)
        super().__init__(ideal=ideal, nadir=nadir, **kwargs)
        self.pf = self.normalization.forward(pf)

    def _do(self, F):
        return self._calc(F, self.pf)

    def _calc(self, F, pf):
        raise NotImplementedError


class Epsilon(EpsilonIndicator):

    def _calc(self, F, pf):
        return _epsilon_additive(F, ref=pf)


class EpsilonMultiplicative(EpsilonIndicator):

    def _calc(self, F, pf):
        return _epsilon_mult(F, ref=pf)
