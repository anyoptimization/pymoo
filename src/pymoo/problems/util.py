import copy
import os
import types

import numpy as np


def decompose(problem, decomposition, weights, return_copy=True):
    if return_copy:
        problem = copy.deepcopy(problem)

    problem._multi_evaluate = problem._evaluate

    def _evaluate(self, x, out, *args, **kwargs):
        self._multi_evaluate(x, out, *args, **kwargs)
        out["F"] = decomposition.do(out["F"], weights, _type="many_to_one")

    problem._evaluate = types.MethodType(_evaluate, problem)
    problem.n_obj = 1

    return problem


def load_pareto_front_from_file(fname):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(current_dir, "pf", "%s" % fname)
    if os.path.isfile(fname):
        pf = np.loadtxt(fname)
        return pf[pf[:, 0].argsort()]


def binomial(n, k):
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0
