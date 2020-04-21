import os
import subprocess

import numpy as np

from pymoo.model.problem import Problem


class MOPTA08(Problem):
    def __init__(self, exec):
        super().__init__(n_var=124, n_obj=1, n_constr=68, xl=0, xu=1, type_var=np.double, elementwise_evaluation=True)
        self.exec = exec

    def _evaluate(self, x, out, *args, **kwargs):
        inputs = os.linesep.join([str(e) for e in x])
        res = subprocess.run(self.exec, input=inputs, text=True, stdout=subprocess.PIPE)
        val = np.array([float(e) for e in res.stdout.split()])
        out["F"] = val[0]
        out["G"] = val[1:]

    def _calc_pareto_front(self, *args, **kwargs):
        return 222.74




