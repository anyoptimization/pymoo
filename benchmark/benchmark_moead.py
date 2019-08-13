import os
import pickle

from benchmark.benchmark_nsga3 import setup
from pymoo.algorithms.moead import moead

if __name__ == '__main__':

    # all the files to be run in a list
    run_files = []

    # prefix of the folder to save the files
    prefix = "runs"

    # name of the experiment
    name = "pymoead"

    # number of runs to execute
    n_runs = 50

    # single to be investigated
    problems = setup.keys()

    # path were the files for this experiment are saved
    path = os.path.join(prefix, name)

    for _problem in problems:

        s = setup[_problem]
        problem = s['problem']

        method = moead(
            s['ref_dirs'],
            n_neighbors=20,
            decomposition='auto',
            prob_neighbor_mating=0.9,
            eliminate_duplicates=True
        )

        termination = s['termination']

        for run in range(1, n_runs+1):

            fname = "%s_%s.run" % (_problem, run)
            _in = os.path.join(path, fname)
            _out = "results/%s/%s/%s_%s.out" % (name, _problem.replace("_", "/"), _problem, run)

            data = {
                'args': [problem, method, termination],
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

