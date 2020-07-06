import os
import pickle

import numpy as np

from pymoo.algorithms.ctaea import CTAEA
from pymoo.factory import get_problem, get_reference_directions
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation


def get_setup(n_obj):
    if n_obj == 3:
        pop_size = 91
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_points=pop_size)

    return {
        'ref_dirs': ref_dirs,
        'crossover': SimulatedBinaryCrossover(20, n_offsprings=1, prob=0.9),
        'mutation': PolynomialMutation(20)
    }


setup = {

    # ==========================================
    # C1-DTLZ1
    # ==========================================

    'c1dtlz1_3obj': {
        'termination': ('n_gen', 500),
        'problem': get_problem("c1dtlz1", None, 3, k=5),
        **get_setup(3)

    },

    # ==========================================
    # C1-DTLZ3
    # ==========================================

    'c1dtlz3_3obj': {
        'termination': ('n_gen', 1000),
        'problem': get_problem("c1dtlz3", None, 3, k=10),
        **get_setup(3)
    },

    # ==========================================
    # C3-DTLZ4
    # ==========================================

    "c3dtlz4_3obj": {
        'termination': ('n_gen', 750),
        'problem': get_problem("c3dtlz4", None, 3, k=10),
        **get_setup(3)
    },

    # ==========================================
    # DC1-DTLZ1
    # ==========================================

    "dc1dtlz1_3obj": {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dc1dtlz1", 12, 3),
        **get_setup(3)
    },

    # ==========================================
    # DC1-DTLZ3
    # ==========================================

    "dc1dtlz3_3obj": {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dc1dtlz3", 12, 3),
        **get_setup(3)
    },

    # ==========================================
    # DC2-DTLZ1
    # ==========================================

    "dc2dtlz1_3obj": {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dc2dtlz1", 12, 3),
        **get_setup(3)
    },

    # ==========================================
    # DC2-DTLZ3
    # ==========================================

    "dc2dtlz3_3obj": {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dc2dtlz3", 12, 3),
        **get_setup(3)
    },

    # ==========================================
    # DC3-DTLZ1
    # ==========================================

    "dc3dtlz1_3obj": {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dc3dtlz1", 12, 3),
        **get_setup(3)
    },

    # ==========================================
    # DC3-DTLZ3
    # ==========================================

    "dc3dtlz3_3obj": {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dc3dtlz3", 12, 3),
        **get_setup(3)
    },

}


if __name__ == '__main__':

    # all the files to be run in a list
    run_files = []

    # prefix of the folder to save the files
    prefix = "runs"

    # name of the experiment
    name = "pyctaea-0.4.2"

    # number of runs to execute
    n_runs = 30

    # single to be investigated
    problems = setup.keys()

    # path were the files for this experiment are saved
    path = os.path.join(prefix, name)

    for _problem in problems:

        s = setup[_problem]
        problem = s['problem']

        method = CTAEA(
            s['ref_dirs'],
            crossover=s['crossover'],
            mutation=s['mutation'],
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
