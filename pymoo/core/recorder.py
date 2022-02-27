from abc import abstractmethod

import numpy as np

from pymoo.core.callback import Callback
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus


class Recorder(Callback):

    def __init__(self, nth_evals=None) -> None:
        super().__init__()
        self.data = []
        self.nth_evals = nth_evals
        self.rec_n_evals = 0

    def notify(self, algorithm, **kwargs):

        if self.nth_evals is None:
            self.data.append(self.save(algorithm))

        else:
            n_evals = algorithm.evaluator.n_eval

            if n_evals >= self.rec_n_evals:
                self.data.append(self.save(algorithm))
                self.rec_n_evals = (1 + (n_evals // self.nth_evals)) * self.nth_evals

    @abstractmethod
    def save(self, algorithm):
        pass

    def get(self, *args, as_array=True):
        ret = []
        for arg in args:
            e = [entry.get(arg) for entry in self.data]
            if as_array:
                e = np.array(e)
            ret.append(e)
        return tuple(ret)


class DefaultSingleObjectiveRecorder(Recorder):

    def save(self, algorithm):
        n_evals = algorithm.evaluator.n_eval

        opt = algorithm.opt
        _feas, _cv, _f = opt.get("feas", "cv", "f")

        cv = _cv.min()

        if np.any(_feas):
            f = _f[_feas].min()
        else:
            f = np.inf

        fgap = np.inf
        try:
            pf = algorithm.problem.pareto_front()
        except:
            pf = None

        if pf is not None:
            fgap = f - pf.min()

        return dict(n_evals=n_evals, cv=cv, f=f, fgap=fgap)


class DefaultMultiObjectiveRecorder(Recorder):

    def save(self, algorithm):

        igd, igd_plus = np.inf, np.inf

        opt = algorithm.opt

        # get all optimal solutions that are feasible
        feas_opt = opt[opt.get("feas")]

        if len(feas_opt) > 0:

            try:
                pf = algorithm.problem.pareto_front()
            except:
                pf = None

            if pf is not None:
                F = feas_opt.get("F")
                igd = IGD(pf=pf, zero_to_one=True).do(F)
                igd_plus = IGDPlus(pf=pf, zero_to_one=True).do(F)

        return {
            "n_evals": algorithm.evaluator.n_eval,
            "cv": opt.get("cv").min(),
            "igd": igd,
            "igd+": igd_plus,
        }
