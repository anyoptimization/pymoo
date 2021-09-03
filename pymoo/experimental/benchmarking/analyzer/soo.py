import numpy as np

from pymoo.experimental.benchmarking.analyzer.analyzer import Analyzer
from pymoo.util.misc import from_dict


class SingleObjectiveAnalyzer(Analyzer):

    def do(self, data, benchmark=None, inplace=False, **kwargs):
        assert benchmark is not None, "The benchmark is necessary to retrieve the known optimum of a funtion"

        problem = benchmark.problems[data["problem"]]["obj"]
        CV, F = from_dict(data, "CV", "F")

        f = F[0, 0]
        cv = CV[0, 0]

        fopt = problem.pareto_front()
        if fopt is not None:
            fopt = fopt[0, 0].astype(np.float)

        fgap = f - fopt

        ret = {
            "f": f,
            "cv": cv,
            "fopt": fopt,
            "fgap": fgap
        }

        if inplace:
            for k, v in ret.items():
                data[k] = v

        return ret



