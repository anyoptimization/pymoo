from math import log

from pymoo.core.termination import Termination


class IndicatorTermination(Termination):

    def __init__(self, indicator, threshold, goal, **kwargs) -> None:
        super().__init__()

        # the indicator to be used
        self.indicator = indicator

        # define the threshold for termination
        self.threshold = threshold

        # what is the optimization goal for this indicator
        self.goal = goal
        assert goal in ["minimize", "maximize"]

        # optional parameters when the indicator calculation is performed
        self.kwargs = kwargs

        # initial the minimum and maximum values of the indicator
        self._min = float("inf")
        self._max = -float("inf")

    def _update(self, algorithm):

        # get the objective space values
        F = algorithm.opt.get("F")

        # get the resulting value from the indicator
        v = self.indicator.do(F, **self.kwargs)

        threshold = self.threshold

        # update the minimum and maximum boundary ranges
        self._min = min(self._min, v)
        self._max = max(self._max, v)
        _min, _max = self._min, self._max

        # depending on the goal either set the percentage
        if self.goal == "minimize":
            perc = 1 - (v - threshold) / (_max - threshold)
        else:
            perc = (v - _min) / (threshold - _min)

        return perc
