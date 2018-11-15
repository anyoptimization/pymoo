"""
This is the experiment for nsga2.
"""
import os
import pickle

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymop.factory import get_problem

setup = {

    'zdt1': {
        'pop_size': 100,
        'termination': ('n_gen', 200),
        'problem': get_problem("zdt1", n_var=30),
        'crossover': SimulatedBinaryCrossover(0.9, 15),
        'mutation': PolynomialMutation(20),
    },
    'zdt2': {
        'pop_size': 100,
        'termination': ('n_gen', 200),
        'problem': get_problem("zdt2", n_var=30),
        'crossover': SimulatedBinaryCrossover(0.9, 15),
        'mutation': PolynomialMutation(20)

    },
    'zdt3': {
        'pop_size': 100,
        'termination': ('n_gen', 200),
        'problem': get_problem("zdt3", n_var=30),
        'crossover': SimulatedBinaryCrossover(0.9, 15),
        'mutation': PolynomialMutation(20)

    },
    'zdt4': {
        'pop_size': 100,
        'termination': ('n_gen', 200),
        'problem': get_problem("zdt4", n_var=10),
        'crossover': SimulatedBinaryCrossover(0.9, 15),
        'mutation': PolynomialMutation(20)
    },
    'zdt6': {
        'pop_size': 100,
        'termination': ('n_gen', 400),
        'problem': get_problem("zdt6", n_var=10),
        'crossover': SimulatedBinaryCrossover(0.9, 15),
        'mutation': PolynomialMutation(20)
    },
    'kursawe': {
        'pop_size': 100,
        'termination': ('n_gen', 100),
        'problem': get_problem("kursawe"),
        'crossover': SimulatedBinaryCrossover(0.9, 10),
        'mutation': PolynomialMutation(10)
    },
    'bnh': {
        'pop_size': 100,
        'termination': ('n_gen', 150),
        'problem': get_problem("bnh"),
        'crossover': SimulatedBinaryCrossover(0.9, 10),
        'mutation': PolynomialMutation(20)
    }
    ,
    'osy': {
        'pop_size': 200,
        'termination': ('n_gen', 250),
        'problem': get_problem("osy"),
        'crossover': SimulatedBinaryCrossover(0.9, 5),
        'mutation': PolynomialMutation(5)
    }

}

if __name__ == '__main__':

    test = get_problem("bnh")
    print(test.pareto_front())

    method = "nsga2"
    n_runs = 100
    #problems = ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']
    problems = setup.keys()

    for e in problems:

        s = setup[e]
        problem = s['problem']

        for run in range(n_runs):
            data = {
                'args': [problem, "nsga2"],
                'kwargs': {
                    'method_args': {
                        'pop_size': s['pop_size'],
                        'crossover': s['crossover'],
                        'mutation': s['mutation'],
                    },
                    'termination': s['termination']
                },
                'out': "%s/pynsga2_%s_%s.out" % (e, e, (run + 1)),
            }

            fname = "pynsga2_%s_%s.run" % (e, (run + 1))
            with open(os.path.join("pynsga2", fname), 'wb') as f:
                pickle.dump(data, f)
