import numpy as np

from pymoo.experimental.benchmarking.analyzer.analyzer import Analyzer
from pymoo.experimental.benchmarking.util import fill_forward_if_nan


class ConvergenceAnalyzer(Analyzer):

    def __init__(self, nan_if_not_available=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nan_if_not_available = nan_if_not_available

    def do(self, data, **kwargs):
        t = []
        for entry in data:
            t.extend([e["n_evals"] for e in entry["callback"]])
        t = sorted(list(set(t)))

        hash = {}
        for k, v in enumerate(t):
            hash[v] = k

        for i, entry in enumerate(data):

            vals = np.full(len(t), np.nan)

            for e in entry["callback"]:
                vals[hash[e["n_evals"]]] = e["f"]

            fill_forward_if_nan(vals)

            entry["n_evals"] = t
            entry["conv"] = vals
            if "fopt" in entry:
                entry["conv_gap"] = vals - entry["fopt"]
