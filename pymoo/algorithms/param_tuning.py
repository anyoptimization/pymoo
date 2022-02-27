import json
import os
import pickle
from os.path import dirname

try:
    import optuna
    from optuna.samplers import TPESampler
except:
    raise Exception("Please install optuna: pip install -U optuna")

from copy import deepcopy

import numpy as np

from pymoo.core.parameters import get_params, set_params, hierarchical, flatten
from pymoo.core.variable import Real, Choice, Binary, Integer
from pymoo.optimize import minimize


class ParameterTuning:

    def __init__(self,
                 n_max_trials,
                 performance,
                 problem,
                 algorithm,
                 termination,
                 params=None,
                 extensively=False,
                 verbose=True,
                 seed=1,
                 pkl=None,
                 json=None) -> None:

        super().__init__()

        # the problem what is the tuning for
        self.problem = problem

        # the optimization method we are trying to find good parameters for
        self.algorithm = algorithm

        # how long an algorithm should be run
        self.termination = termination

        # the exact parameters we are optimizing
        if params is None:
            params = get_params(algorithm)
        self.params = params

        # the criterion to check the performance of a method
        self.performance = performance

        # the maximum number of trials in total
        self.n_max_trials = n_max_trials

        # the number of trials already being executed
        self.n_trials = 0

        # if provided files are written: pkl (the whole object), json (the current best parameter configuration)
        self.pkl = pkl
        self.json = json

        # whether the whole search space or just the one that is recommended should be used
        self.extensively = extensively

        # the optuna study used to find parameters
        self.study = optuna.create_study(study_name="Parameter Tuning", sampler=TPESampler(seed=seed))

        # store each run here as a list if it needs to be accessed later
        self.trials = []

        # whether output should be printed or not
        self.verbose = verbose

    def has_next(self):
        return self.n_trials < self.n_max_trials

    def best(self):
        study = self.study

        best = hierarchical(study.best_params)
        algorithm = deepcopy(self.algorithm)
        set_params(algorithm, best)

        return dict(algoritm=algorithm, performance=study.best_trial.value, params=best, study=study)

    def do(self):

        # while there is another trial to try
        while self.has_next():
            # do the next iteration
            self.next()

        # and finally return the best
        return self.best()

    def next(self):
        problem, algorithm, termination, study, params = self.problem, self.algorithm, self.termination, self.study, self.params

        # ask for a new trial
        trial = study.ask()

        # create a copy of the algorithm to keep the original unmodified
        method = deepcopy(algorithm)

        # ask optuna what parameter configuration should be used next
        vals = {}
        for name, param in flatten(params).items():
            if isinstance(param, Real):
                lower, upper = param.bounds if not self.extensively else param.strict
                v = trial.suggest_float(name, lower, upper)
            elif isinstance(param, Integer):
                lower, upper = param.bounds if not self.extensively else param.strict
                v = trial.suggest_int(name, lower, upper)
            elif isinstance(param, Choice):
                options = param.options if not self.extensively else param.all
                v = trial.suggest_categorical(name, options)
            elif isinstance(param, Binary):
                v = trial.suggest_categorical(name, [False, True])
            else:
                raise Exception("Type not supported yet.")
            vals[name] = v

        # set them to the copied method
        set_params(method, hierarchical(vals))

        # evaluate the performance of the algorithm
        ret = self.performance.evaluate(problem, method, termination=termination)

        # let optuna know the result
        study.tell(trial, ret["value"])

        if self.verbose:
            print(self.n_trials, ret["value"], ret.get("display"), vals)

        # and add the trial to do so bookkeeping
        self.trials.append(ret)

        # increase the iteration counter
        self.n_trials += 1

        if self.pkl is not None:
            folder = dirname(self.pkl)
            if len(folder) > 0:
                if not os.path.exists(folder):
                    os.makedirs(folder)
            with open(self.pkl, 'wb') as f:
                pickle.dump(self, f)

        if self.json is not None:
            folder = dirname(self.json)
            if len(folder) > 0:
                if not os.path.exists(folder):
                    os.makedirs(folder)
            with open(self.json, 'w') as f:
                json.dump(hierarchical(study.best_params), f)


class Performance:

    def __init__(self, n_runs=None, seed=1) -> None:
        super().__init__()
        self.n_runs = n_runs

        np.random.seed(seed)
        self.seeds = np.random.randint(1, 1000000, size=self.n_runs)

    def evaluate(self, problem, algorithm, **kwargs):
        pass


class SimpleSingleObjectivePerformance(Performance):

    def evaluate(self, problem, algorithm, **kwargs):
        runs = [minimize(problem, algorithm, seed=seed, **kwargs) for seed in self.seeds]
        F = np.array([run.F.min() for run in runs])
        return dict(value=F.mean(), std=F.std(), as_list=F, display=dict(std=F.std()))


class ConvergenceSingleObjectivePerformance(Performance):

    def __init__(self, pf, tol=1e-8, n_runs=None, seeds=None) -> None:
        super().__init__(n_runs, seeds)
        self.fmin = pf.min()
        self.tol = tol

    def evaluate(self, problem, algorithm, **kwargs):

        F = []
        runs = []

        for seed in self.seeds:
            n_evals, fgap = [], []

            method = deepcopy(algorithm)
            method.setup(problem, seed=seed, **kwargs)

            while method.has_next():
                method.next()

                n_evals.append(method.evaluator.n_eval)

                v = method.opt.get("F").min() - self.fmin
                fgap.append(v)

                if self.tol is not None:
                    if v < self.tol:
                        break

            integral = stepwise_integral(n_evals, fgap)

            F.append(integral)
            runs.append(method.result())

        avg_fgap = np.array([run.F.min() - self.fmin for run in runs]).mean()

        return dict(value=np.mean(F), std=np.std(F), each=np.array(F), runs=runs, display=dict(avg_fgap=avg_fgap))


def stepwise_integral(n_evals, val):
    return n_evals[0] * val[0] + (np.diff(n_evals) * np.array(val)[1:]).sum()

# def moo_performance(problem, algorithm, seeds, **kwargs):
#     assert problem.n_constr == 0, "This is only implemented for unconstrained problem at this point."
#
#     pf = problem.pareto_front()
#     perf = lambda F: IGD(pf).do(F)
#
#     class MyCallback(Callback):
#
#         def __init__(self) -> None:
#             super().__init__()
#             self.data = []
#
#         def notify(self, algorithm, **kwargs):
#             if algorithm.n_gen % 10 == 0:
#                 igd = perf(algorithm.opt.get("F"))
#                 self.data.append(igd)
#
#     f = []
#     ret = []
#     for seed in seeds:
#         callback = MyCallback()
#         res = minimize(problem, algorithm, seed=seed, callback=callback, **kwargs)
#
#         data = callback.data
#         data.append(perf(res.F))
#
#         avg_igd = np.mean(data)
#
#         f.append(avg_igd)
#         ret.append(res)
#     f = np.array(f)
#     return ret, f.mean(), f.std()
