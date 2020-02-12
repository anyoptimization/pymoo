from pymoo.performance_indicator.igd import IGD
from pymoo.util.normalization import normalize
from pymoo.util.termination.tolerance import ToleranceBasedTermination
import numpy as np


class MultiObjectiveSpaceToleranceTermination(ToleranceBasedTermination):

    def __init__(self,
                 renormalize=False,
                 all_to_current=False,
                 **kwargs) -> None:
        """

        Terminate based on the objective space movement during a run of a multi-objective optimization algorithm

        Parameters
        ----------
        renormalize : bool
            To be more precisely in the difference calculation all n_last point sets need to be renormalized.
            However, since it only terminates when the ideal and nadir point are not moving significantly anymore
            this is by default disabled.

        all_to_current : bool
            (Only if renormalize is enabled) Calculate the movement with respect to the current population
            instead of always from last to current for each transition.

        kwargs
        """
        super().__init__(n_hist_at_least=2, **kwargs)
        self.renormalize = renormalize
        self.all_to_current = all_to_current

    def _store(self, algorithm):
        F = algorithm.opt.get("F")

        return {
            "ideal": F.min(axis=0),
            "nadir": F.max(axis=0),
            "F": F
        }

    def _calc_delta(self, a, b):
        return np.max(np.abs((a - b)))

    def _calc_delta_norm_old(self, a, b):
        return np.max(np.abs((a - b)) / np.abs((a + b) / 2))

    def _calc_delta_norm(self, a, b, norm):
        return np.max(np.abs((a - b) / norm))

    def _calc_metric(self):

        # get the current and the last history snapshot
        current, last = self.history[-1], self.history[-2]

        # this is the range between the nadir and the ideal point
        norm = current["nadir"] - current["ideal"]
        # if the range is degenerated (very close to zero) - disable normalization by dividing by one
        norm[norm < 1e-32] = 1

        # calculate the change from last to current in ideal and nadir point
        delta_ideal = self._calc_delta_norm(current["ideal"], last["ideal"], norm)
        max_delta_ideal = max([e["delta_ideal"] for e in self.metrics] + [delta_ideal])

        delta_nadir = self._calc_delta_norm(current["nadir"], last["nadir"], norm)
        max_delta_nadir = max([e["delta_nadir"] for e in self.metrics] + [delta_nadir])

        # get necessary data from the current population
        c_F, c_ideal, c_nadir = current["F"], current["ideal"], current["nadir"]
        c_N = normalize(c_F, c_ideal, c_nadir)

        if not self.renormalize:
            l_N = normalize(last["F"], c_ideal, c_nadir)
            delta_f = IGD(c_N).calc(l_N)
            max_delta_f = max([e["delta_f"] for e in self.metrics] + [delta_f])

        else:
            # normalize all previous generations with respect to current ideal and nadir
            N = [normalize(e["F"], c_ideal, c_nadir) for e in self.history]

            # check if the movement of all points is significant
            if self.all_to_current:
                delta_f = [IGD(c_N).calc(N[k]) for k in range(len(N))]
            else:
                delta_f = [IGD(N[k + 1]).calc(N[k]) for k in range(len(N) - 1)]

            max_delta_f = np.array(delta_f).max()

        return {
            "delta_ideal": delta_ideal,
            "max_delta_ideal": max_delta_ideal,
            "delta_nadir": delta_nadir,
            "max_delta_nadir": max_delta_nadir,
            "delta_f": delta_f,
            "max_delta_f": max_delta_f,
            "max_delta_all": max(max_delta_ideal, max_delta_nadir, max_delta_f)
        }

    def _decide(self):
        return self.metrics[-1]["max_delta_all"] > self.tol


class SingleObjectiveSpaceToleranceTermination(ToleranceBasedTermination):

    def __init__(self,
                 tol=1e-6,
                 tol_rel=0.0001,
                 **kwargs) -> None:
        super().__init__(n_hist_at_least=2, tol=tol, **kwargs)
        self.tol_rel = tol_rel

    def _store(self, algorithm):
        return algorithm.opt.get("F").min()

    def _calc_metric(self):
        # get the current and the last history snapshot
        current, last = self.history[-1], self.history[-2]
        delta_f = last - current

        return {
            "f": current,
            "delta_f": delta_f
        }

    def _decide(self):
        F = [e["f"] for e in self.metrics]
        _min = min(F)
        if _min == 0:
            _min = 1e-64

        max_delta_f = max([e["delta_f"] for e in self.metrics])
        max_delta_f_rel = max([(e["f"] / _min) - 1 for e in self.metrics])

        return max_delta_f > self.tol or max_delta_f_rel > self.tol_rel
