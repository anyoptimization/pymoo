"""
This is the experiment for nsga2.
"""
import os
import pickle

from pymoo.algorithms.nsga2 import NSGA2

from pymoo.factory import get_sampling, get_crossover, get_mutation, get_problem
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation

setup = {

    'zdt1': {
        'pop_size': 100,
        'termination': ('n_gen', 200),
        'problem': get_problem("zdt1", n_var=30),
        'crossover': SimulatedBinaryCrossover(15),
        'mutation': PolynomialMutation(20),
    },
    'zdt2': {
        'pop_size': 100,
        'termination': ('n_gen', 200),
        'problem': get_problem("zdt2", n_var=30),
        'crossover': SimulatedBinaryCrossover(15),
        'mutation': PolynomialMutation(20)

    },
    'zdt3': {
        'pop_size': 100,
        'termination': ('n_gen', 200),
        'problem': get_problem("zdt3", n_var=30),
        'crossover': SimulatedBinaryCrossover(15),
        'mutation': PolynomialMutation(20)

    },
    'zdt4': {
        'pop_size': 100,
        'termination': ('n_gen', 200),
        'problem': get_problem("zdt4", n_var=10),
        'crossover': SimulatedBinaryCrossover(15),
        'mutation': PolynomialMutation(20)
    },
    'zdt5': {
        'pop_size': 100,
        'termination': ('n_gen', 400),
        'problem': get_problem("zdt5", normalize=False),
        'sampling': get_sampling("bin_random"),
        'crossover': get_crossover("bin_two_point"),
        'mutation': get_mutation("bin_bitflip")
    },
    'zdt6': {
        'pop_size': 100,
        'termination': ('n_gen', 400),
        'problem': get_problem("zdt6", n_var=10),
        'crossover': SimulatedBinaryCrossover(15),
        'mutation': PolynomialMutation(20)
    },
    'kursawe': {
        'pop_size': 100,
        'termination': ('n_gen', 100),
        'problem': get_problem("kursawe"),
        'crossover': SimulatedBinaryCrossover(10),
        'mutation': PolynomialMutation(10)
    },
    'bnh': {
        'pop_size': 100,
        'termination': ('n_gen', 150),
        'problem': get_problem("bnh"),
        'crossover': SimulatedBinaryCrossover(10),
        'mutation': PolynomialMutation(20)
    }
    ,
    'osy': {
        'pop_size': 200,
        'termination': ('n_gen', 250),
        'problem': get_problem("osy"),
        'crossover': SimulatedBinaryCrossover(5),
        'mutation': PolynomialMutation(5)
    }

}

if __name__ == '__main__':

    # all the files to be run in a list
    run_files = []

    # prefix of the folder to save the files
    prefix = "runs"

    # name of the experiment
    name = "pynsga2-0.4.0"

    # number of runs to execute
    n_runs = 100

    # single to be investigated
    #problems = ['zdt5']
    problems = setup.keys()

    # path were the files for this experiment are saved
    path = os.path.join(prefix, name)

    for _problem in problems:

        s = setup[_problem]
        problem = s['problem']

        method = NSGA2(
            pop_size=s['pop_size'],
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

