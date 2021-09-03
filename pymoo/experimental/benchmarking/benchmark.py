import json
import os
import pickle
from copy import deepcopy
from math import ceil

import numpy as np

from pymoo.experimental.benchmarking.util import at_least2d
from pymoo.optimize import minimize
from pymoo.util.misc import from_dict


class Benchmark:

    def __init__(self,
                 recorder=None,
                 n_runs=11,
                 default_termination=None):
        super().__init__()

        self.default_callback = recorder
        self.default_runs = n_runs
        self.default_termination = default_termination

        self.writer = None
        self.loader = None
        self.extractor = None

        self.problems = {}
        self.runs = []

        self.results = []

    def add_problem(self, label, problem, termination=None):
        self.problems[label] = dict(label=label, obj=problem, termination=termination)

    def add_algorithm(self, label, algorithm, problems=None, n_runs=None, termination=None, callback=None):

        if callback is None:
            callback = self.default_callback

        if problems is None:

            if len(self.problems) == 0:
                raise Exception("No Problem have been added to run the algorithm on!")

            problems = list(self.problems.keys())
        elif not isinstance(problems, list):
            problems = [problems]

        if n_runs is None:
            n_runs = self.default_runs

        for problem in problems:

            if termination is None:
                termination = self.problems[problem]["termination"]

            if termination is None:
                termination = self.default_termination

            for run in range(1, n_runs + 1):
                args = dict(problem=self.problems[problem]["obj"], algorithm=algorithm, termination=termination,
                            seed=run, callback=callback)
                e = dict(problem=problem, algorithm=label, run=run, args=args)
                self.runs.append(e)

    def run(self, ordered_by="problem", batch=None, verbose=True, **kwargs):

        params = self.runs
        params.sort(key=lambda x: x[ordered_by])

        if batch is not None:
            batch, n_batches = [int(e) for e in batch.split("/")]
            assert 0 < batch <= n_batches, f"Batch number must be greater than 0 but less or equal to number of batches!"

            batch_size = ceil(len(params) / n_batches)
            i, j = (batch - 1) * batch_size, batch * batch_size
            params = params[i:j]

        if "extractor" not in kwargs:
            kwargs["extractor"] = DefaultExtractor()

        self.results = run_looped(params, batch=batch, verbose=verbose, **kwargs)
        # self.results = run_parallel(kwargs.get("starmap"), self, params, batch=batch, verbose=verbose, **kwargs)

        return self.results


# ---------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------


def run_looped(params,
               batch,
               verbose=False,
               loader=None,
               writer=None,
               extractor=None,
               mem_free=False,
               run_if_loading_fails=True,
               exception_if_not_available=False,
               **kwargs):
    ret = []
    for i, param in enumerate(params):

        problem, algorithm, run, args = from_dict(param, "problem", "algorithm", "run", "args")

        if verbose:
            line = f"{i + 1}/{len(params)} | {algorithm} | {problem} | {run}"
            if batch is not None:
                line = f"{batch} | {line}"
            print(line, end="")

        entry = None

        if loader is not None:
            entry = loader.load(param)
            if entry is not None:
                if verbose:
                    print(f" | Loaded")

        if entry is None:
            if run_if_loading_fails:

                res = execute(param)

                entry = extractor.extract(param, res)

                if verbose:
                    print(f" | {np.round(res.exec_time, 6)} s")

                if writer is not None:
                    writer.write(entry)

            else:
                if exception_if_not_available:
                    assert entry is not None, f"Error while loading {param}"
                if verbose:
                    print(f" | Failed")

        # if the algorithm should not keep anything in memory and just write files do that
        if entry is not None and not mem_free:
            ret.append(entry)

    return ret


def execute(param):
    args = deepcopy(param['args'])

    problem = args.pop("problem")
    algorithm = args.pop("algorithm")
    res = minimize(problem, algorithm, return_least_infeasible=True, **args)
    return res


class IO:

    def __init__(self, folder) -> None:
        super().__init__()
        self.folder = folder


class DefaultWriter(IO):

    def write(self, entry):

        problem, algorithm, run, callback = from_dict(entry, "problem", "algorithm", "run", "callback")

        folder = os.path.join(self.folder, algorithm, problem)
        if not os.path.exists(folder):
            os.makedirs(folder)

        for key in ["X", "CV", "F"]:
            np.savetxt(os.path.join(folder, f"{run}.{key.lower()}"), entry.get(key))

        path = os.path.join(folder, f"{run}.json")
        with open(path, 'w') as f:
            json.dump(entry.get("info"), f, ensure_ascii=False, indent=4)

        if callback is not None:
            pickle.dump(callback, open(os.path.join(folder, f"{run}.dat"), 'wb'))


class DefaultLoader(IO):

    def load(self, entry):

        problem, algorithm, run = from_dict(entry, "problem", "algorithm", "run")

        entry = dict(entry)
        del entry["args"]

        path = os.path.join(self.folder, algorithm, problem)
        if os.path.exists(path):

            for key in ["X", "CV", "F"]:
                file = os.path.join(path, f"{run}.{key.lower()}")
                if os.path.exists(file):
                    try:
                        vals = np.loadtxt(file)
                    except:
                        return None
                    if len(vals.shape) == 0:
                        vals = np.array([float(vals)])
                    entry[key] = at_least2d(vals, expand="r")
                else:
                    return None

            file = os.path.join(path, f"{run}.dat")
            if os.path.exists(file):
                entry["callback"] = pickle.load(open(file, 'rb'))

            return entry


class DefaultExtractor:

    def extract(self, param, res):
        opt = res.opt

        entry = {
            "problem": param["problem"],
            "algorithm": param["algorithm"],
            "run": param["run"],
            "X": opt.get("X"),
            "CV": opt.get("CV"),
            "F": opt.get("F"),
            "info": dict(time=res.exec_time),
        }

        if res.algorithm.callback is not None:
            entry["callback"] = res.algorithm.callback.data,

        return entry
