import numpy as np

from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.util.normalization import normalize
from pymoo.termination.delta import DeltaToleranceTermination


def calc_delta(a, b):
    return np.max(np.abs((a - b)))


def calc_delta_norm(a, b, norm):
    return np.max(np.abs((a - b) / norm))


class SingleObjectiveSpaceTermination(DeltaToleranceTermination):

    def __init__(self, tol=1e-6, only_feas=True, **kwargs) -> None:
        super().__init__(tol, **kwargs)
        self.only_feas = only_feas

    def _delta(self, prev, current):
        if prev == np.inf or current == np.inf:
            return np.inf
        else:
            return max(0, prev - current)

    def _data(self, algorithm):
        opt = algorithm.opt
        f = opt.get("f")

        if self.only_feas:
            f = f[opt.get("feas")]

        if len(f) > 0:
            return f.min()
        else:
            return np.inf


class MultiObjectiveSpaceTermination(DeltaToleranceTermination):

    def __init__(self, tol=0.0025, only_feas=True, **kwargs):
        super().__init__(tol, **kwargs)
        self.delta_ideal = None
        self.delta_nadir = None
        self.delta_f = None
        self.only_feas = only_feas

    def _data(self, algorithm):
        feas, F = algorithm.opt.get("feas", "F")

        if self.only_feas:
            F = F[feas]

        if len(F) > 0:
            return dict(ideal=F.min(axis=0), nadir=F.max(axis=0), F=F, feas=True)
        else:
            return dict(ideal=None, nadir=None, F=F, feas=False)

    def _delta(self, prev, current):

        if not (prev["feas"] and current["feas"]):
            return np.inf

        # this is the range between the nadir and the ideal point
        norm = current["nadir"] - current["ideal"]

        # if the range is degenerated (very close to zero) - disable normalization by dividing by one
        norm[norm < 1e-32] = 1.0

        # calculate the change from last to current in ideal and nadir point
        delta_ideal = calc_delta_norm(current["ideal"], prev["ideal"], norm)
        delta_nadir = calc_delta_norm(current["nadir"], prev["nadir"], norm)

        # get necessary data from the current population
        c_F, c_ideal, c_nadir = current["F"], current["ideal"], current["nadir"]
        p_F = prev["F"]

        # normalize last and current with respect to most recent ideal and nadir
        c_N = normalize(c_F, c_ideal, c_nadir)
        p_N = normalize(p_F, c_ideal, c_nadir)

        # calculate IGD from one to another
        delta_f = IGD(c_N).do(p_N)

        # store the delta values to the object
        self.delta_ideal, self.delta_nadir, self.delta_f = delta_ideal, delta_nadir, delta_f

        return max(delta_ideal, delta_nadir, delta_f)


class MultiObjectiveSpaceTerminationWithRenormalization(MultiObjectiveSpaceTermination):

    def __init__(self,
                 n=30,
                 all_to_current=False,
                 sliding_window=True,
                 indicator="igd",
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.n = n
        self.all_to_current = all_to_current
        self.sliding_window = sliding_window
        self.indicator = indicator

        self.data = []

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
            if self.indicator == "igd":
                delta_f = [IGD(c_N).do(N[k]) for k in range(len(N))]
            elif self.indicator == "hv":
                hv = Hypervolume(ref_point=np.ones(c_F.shape[1]))
                delta_f = [hv.do(N[k]) for k in range(len(N))]
        else:
            delta_f = [IGD(N[k + 1]).do(N[k]) for k in range(len(N) - 1)]

        ret["delta_f"] = delta_f

        return ret

    def _decide(self, metrics):
        delta_ideal = [e["delta_ideal"] for e in metrics]
        delta_nadir = [e["delta_nadir"] for e in metrics]
        delta_f = [max(e["delta_f"]) for e in metrics]
        return max(max(delta_ideal), max(delta_nadir), max(delta_f)) > self.tol
