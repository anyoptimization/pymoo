import numpy as np

from pymoo.experimental.benchmarking.analyzer.analyzer import Analyzer
from pymoo.indicators.igd import IGD
from pymoo.util.misc import from_dict


class MultiObjectiveAnalyzer(Analyzer):

    def do(self, data, scope=None, benchmark=None, inplace=False, **kwargs):
        assert benchmark is not None, "The benchmark is necessary to retrieve the known optimum of a function"

        problem = benchmark.problems[data["problem"]]["obj"]
        CV, F = from_dict(data, "CV", "F")

        igd = np.inf
        pf = problem.pareto_front(**kwargs)
        if pf is not None:
            igd = IGD(pf, zero_to_one=True).do(F)

        ret = {
            "pf": pf,
            "igd": igd,
        }

        if inplace:
            for k, v in ret.items():
                data[k] = v

        return ret




