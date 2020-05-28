import numpy as np

from pymoo.performance_indicator.hv import Hypervolume
from pymoo.performance_indicator.igd import IGD
from pymoo.performance_indicator.igd_plus import IGDPlus
from pymoo.util.normalization import normalize
from pymoo.util.termination.sliding_window_termination import SlidingWindowTermination


def calc_delta(a, b):
    return np.max(np.abs((a - b)))


def calc_delta_norm_old(a, b):
    return np.max(np.abs((a - b)) / np.abs((a + b) / 2))


def calc_delta_norm(a, b, norm):
    return np.max(np.abs((a - b) / norm))


class MultiObjectiveSpaceToleranceTermination(SlidingWindowTermination):

    def __init__(self,
                 tol=0.0025,
                 n_last=30,
                 nth_gen=5,
                 n_max_gen=None,
                 n_max_evals=None,
                 **kwargs) -> None:
        super().__init__(metric_window_size=n_last,
                         data_window_size=2,
                         min_data_for_metric=2,
                         nth_gen=nth_gen,
                         n_max_gen=n_max_gen,
                         n_max_evals=n_max_evals,
                         **kwargs)
        self.tol = tol

    def _store(self, algorithm):
        F = algorithm.opt.get("F")
        return {
            "ideal": F.min(axis=0),
            "nadir": F.max(axis=0),
            "F": F
        }

    def _metric(self, data):
        last, current = data[-2], data[-1]

        # this is the range between the nadir and the ideal point
        norm = current["nadir"] - current["ideal"]

        # if the range is degenerated (very close to zero) - disable normalization by dividing by one
        norm[norm < 1e-32] = 1

        # calculate the change from last to current in ideal and nadir point
        delta_ideal = calc_delta_norm(current["ideal"], last["ideal"], norm)
        delta_nadir = calc_delta_norm(current["nadir"], last["nadir"], norm)

        # get necessary data from the current population
        c_F, c_ideal, c_nadir = current["F"], current["ideal"], current["nadir"]

        # normalize last and current with respect to most recent ideal and nadir
        c_N = normalize(c_F, c_ideal, c_nadir)
        l_N = normalize(last["F"], c_ideal, c_nadir)

        # calculate IGD from one to another
        delta_f = IGD(c_N).calc(l_N)

        return {
            "delta_ideal": delta_ideal,
            "delta_nadir": delta_nadir,
            "delta_f": delta_f
        }

    def _decide(self, metrics):
        delta_ideal = [e["delta_ideal"] for e in metrics]
        delta_nadir = [e["delta_nadir"] for e in metrics]
        delta_f = [e["delta_f"] for e in metrics]
        return max(max(delta_ideal), max(delta_nadir), max(delta_f)) > self.tol


class MultiObjectiveSpaceToleranceTerminationWithRenormalization(MultiObjectiveSpaceToleranceTermination):

    def __init__(self,
                 n_last=30,
                 all_to_current=False,
                 sliding_window=True,
                 perf_indicator="igd",
                 **kwargs) -> None:

        super().__init__(n_last=n_last,
                         truncate_metrics=False,
                         truncate_data=False,
                         **kwargs)
        self.data = []
        self.all_to_current = all_to_current
        self.sliding_window = sliding_window
        self.perf_indicator = perf_indicator

    def _metric(self, data):
        ret = super()._metric(data)

        if not self.sliding_window:
            data = self.data[-self.metric_window_size:]

        # get necessary data from the current population
        current = data[-1]
        c_F, c_ideal, c_nadir = current["F"], current["ideal"], current["nadir"]

        # normalize all previous generations with respect to current ideal and nadir
        N = [normalize(e["F"], c_ideal, c_nadir) for e in data]

        # check if the movement of all points is significant
        if self.all_to_current:
            c_N = normalize(c_F, c_ideal, c_nadir)
            if self.perf_indicator == "igd":
                delta_f = [IGD(c_N).calc(N[k]) for k in range(len(N))]
            elif self.perf_indicator == "hv":
                # delta_f = [IGDPlus(c_N).calc(N[k]) for k in range(len(N))]
                hv = Hypervolume(ref_point=np.ones(c_F.shape[1]))
                delta_f = [hv.calc(N[k]) for k in range(len(N))]
        else:
            delta_f = [IGD(N[k + 1]).calc(N[k]) for k in range(len(N) - 1)]

        ret["delta_f"] = delta_f

        return ret

    def _decide(self, metrics):
        delta_ideal = [e["delta_ideal"] for e in metrics]
        delta_nadir = [e["delta_nadir"] for e in metrics]
        delta_f = [max(e["delta_f"]) for e in metrics]
        return max(max(delta_ideal), max(delta_nadir), max(delta_f)) > self.tol
