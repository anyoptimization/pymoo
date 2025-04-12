from copy import deepcopy

import numpy as np

from pymoo.core.parameters import get_params, flatten, set_params, hierarchical
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


def create(algorithm, params):
    algorithm = deepcopy(algorithm)
    set_params(algorithm, hierarchical(params))
    return algorithm


class HyperparameterProblem(ElementwiseProblem):

    def __init__(self, algorithm, performance, func_create=create, vars=None, **kwargs):

        # get the parameters from the algorithm object
        if vars is None:
            vars = get_params(algorithm)

        if isinstance(vars, dict):
            vars = flatten(vars)

        assert len(vars) > 0, "No hyper-parameters found to optimize."

        super().__init__(vars=vars, **kwargs)

        self.algorithm = algorithm
        self.performance = performance
        self.func_create = func_create

    def _evaluate(self, x, out, *args, **kwargs):
        algorithm = self.func_create(self.algorithm, x)
        v = self.performance(algorithm)
        out.update(v)


class SingleObjectiveSingleRun:

    def __init__(self, problem, **kwargs):
        super().__init__()
        self.problem = problem
        self.kwargs = kwargs

    def __call__(self, algorithm):
        ret = minimize(self.problem, algorithm, **self.kwargs)
        return dict(F=ret.F, G=ret.G, H=ret.H)


def stats_single_objective_mean(rets):
    F, G, H = [], [], []
    for ret in rets:
        F.append(ret.F)
        G.append(ret.G)
        H.append(ret.H)

    F, G, H = np.array(F), np.array(G), np.array(H)

    return dict(F=F.mean(axis=0), G=G.mean(axis=0), H=H.mean(axis=0))


def stats_avg_nevals(rets):
    return dict(F=np.array([ret.algorithm.evaluator.n_eval for ret in rets]).mean())


class MultiRun:

    def __init__(self, problem, n_runs=None, seeds=None, func_stats=stats_single_objective_mean, **kwargs):
        super().__init__()
        self.problem = problem
        self.kwargs = kwargs

        if seeds is None:
            if n_runs is None:
                raise Exception("Either provide number of runs or seeds directly.")

            seeds = np.random.randint(1, 1000000, size=n_runs)

        self.seeds = seeds
        self.func_stats = func_stats

    def __call__(self, algorithm):
        rets = [minimize(self.problem, algorithm, seed=seed, **self.kwargs) for seed in self.seeds]
        out = self.func_stats(rets)
        out["__results__"] = rets
        return out
