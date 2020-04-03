"""
This is the experiment for nsga2.
"""
import os
import pickle

from benchmark.benchmark_nsga2 import setup as setup_nsga2
from benchmark.benchmark_nsga3 import setup as setup_nsga3
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

if __name__ == '__main__':

    # all the files to be run in a list
    run_files = []

    # prefix of the folder to save the files
    prefix = "runs"

    # name of the experiment
    name = "term-0.3.3"

    # number of runs to execute
    n_runs = 51

    # single to be investigated
    for key in setup_nsga2.keys():
        setup_nsga2[key]["algorithm"] = "nsga2"

    for key in setup_nsga3.keys():
        setup_nsga3[key]["algorithm"] = "nsga3"

    # setup = {**setup_nsga2, **setup_nsga3}
    setup = {**setup_nsga3}

    # get the problems
    problems = setup.keys()

    # path were the files for this experiment are saved
    path = os.path.join(prefix, name)

    for _problem in problems:

        s = setup[_problem]
        problem = s['problem']

        if s["algorithm"] == "nsga2":
            method = NSGA2(
                pop_size=s['pop_size'],
                crossover=s['crossover'],
                mutation=s['mutation'],
                eliminate_duplicates=True
            )

        elif s["algorithm"] == "nsga3":
            method = NSGA3(
                s['ref_dirs'],
                pop_size=s['pop_size'],
                crossover=s['crossover'],
                mutation=s['mutation'],
                eliminate_duplicates=True
            )

        for run in range(1, n_runs+1):

            fname = "%s_%s.run" % (_problem, run)
            _in = os.path.join(path, fname)
            _out = "results/%s/%s/%s_%s.out" % (name, _problem.replace("_", "/"), _problem, run)

            data = {
                'args': [problem, method],
                'kwargs': {
                    'seed': run,
                },
                'in': _in,
                'out': _out,
            }

            os.makedirs(os.path.join(os.path.dirname(_in)), exist_ok=True)

            with open(_in, 'wb') as f:
                pickle.dump(data, f)
                run_files.append(data)

        # create the final run.txt file
        with open(os.path.join(prefix, name, "run.bat"), 'w') as f:
            for run_file in run_files:
                f.write("python execute.py %s %s\n" % (run_file['in'], run_file['out']))

