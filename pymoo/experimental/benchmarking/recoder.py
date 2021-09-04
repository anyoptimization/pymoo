from copy import deepcopy

import numpy as np

from pymoo.core.callback import Callback
from pymoo.indicators.igd import IGD


class Recorder(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data = []

    def notify(self, algorithm, **kwargs):
        elem = self.save(algorithm)
        self.data.append(elem)

    def save(self, algorithm):
        pass


class DefaultSingleObjectiveRecorder(Recorder):

    def save(self, algorithm):
        opt = deepcopy(algorithm.opt[0])
        n_evals, cv, f = algorithm.evaluator.n_eval, opt.CV[0], opt.F[0]

        # see if the problem has the best value stored
        pf = algorithm.problem.pareto_front()
        fgap = None

        # if we have on let us measure how close we are
        if pf is not None:
            fopt = pf.min()
            fgap = f - fopt

        return dict(n_evals=n_evals, cv=float(cv), f=float(f), fgap=fgap)


class DefaultMultiObjectiveRecorder(Recorder):

    def save(self, algorithm):

        min_cv = algorithm.opt.get("CV").min()

        igd = None

        pf = algorithm.problem.pareto_front()
        if pf is not None:
            if min_cv <= 0:
                igd = IGD(pf=pf, zero_to_one=True).do(algorithm.opt.get("F"))
            else:
                igd = np.inf

        return {
            "n_evals": algorithm.evaluator.n_eval,
            "cv": min_cv,
            "igd": igd,
        }
