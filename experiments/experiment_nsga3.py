"""
This is the experiment for nsga2.
"""
import os
import pickle

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.util.reference_direction import MultiLayerReferenceDirectionFactory, UniformReferenceDirectionFactory
from pymop.factory import get_problem


def get_setup(n_obj):
    if n_obj == 3:
        pop_size = 92
        ref_dirs = UniformReferenceDirectionFactory(n_obj, n_partitions=12).do()
    elif n_obj == 5:
        pop_size = 212
        ref_dirs = UniformReferenceDirectionFactory(n_obj, n_partitions=6).do()
    elif n_obj == 8:
        pop_size = 156
        ref_dirs = MultiLayerReferenceDirectionFactory([
            UniformReferenceDirectionFactory(n_obj, n_partitions=3, scaling=1.0),
            UniformReferenceDirectionFactory(n_obj, n_partitions=2, scaling=0.5)]).do()
    elif n_obj == 10:
        pop_size = 276
        ref_dirs = MultiLayerReferenceDirectionFactory([
            UniformReferenceDirectionFactory(n_obj, n_partitions=3, scaling=1.0),
            UniformReferenceDirectionFactory(n_obj, n_partitions=2, scaling=0.5)]).do()
    elif n_obj == 15:
        pop_size = 136
        ref_dirs = MultiLayerReferenceDirectionFactory([
            UniformReferenceDirectionFactory(n_obj, n_partitions=2, scaling=1.0),
            UniformReferenceDirectionFactory(n_obj, n_partitions=1, scaling=0.5)]).do()

    return {
        'ref_dirs': ref_dirs,
        'pop_size': pop_size,
        'crossover': SimulatedBinaryCrossover(1.0, 30),
        'mutation': PolynomialMutation(20)
    }


setup = {

    # ==========================================
    # DTLZ1
    # ==========================================

    'dtlz1_3obj': {
        'termination': ('n_gen', 400),
        'problem': get_problem("dtlz1", None, 3, k=5),
        **get_setup(3)

    },

    'dtlz1_5obj': {
        'termination': ('n_gen', 600),
        'problem': get_problem("dtlz1", None, 5, k=5),
        **get_setup(5)
    },

    'dtlz1_8obj': {
        'termination': ('n_gen', 750),
        'problem': get_problem("dtlz1", None, 8, k=5),
        **get_setup(8)
    },

    'dtlz1_10obj': {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dtlz1", None, 10, k=5),
        **get_setup(10)
    },

    'dtlz1_15obj': {
        'termination': ('n_gen', 1500),
        'problem': get_problem("dtlz1", None, 15, k=5),
        **get_setup(15)
    },

    # ==========================================
    # DTLZ2
    # ==========================================

    'dtlz2_3obj': {
        'termination': ('n_gen', 250),
        'problem': get_problem("dtlz2", None, 3, k=10),
        **get_setup(3)

    },

    'dtlz2_5obj': {
        'termination': ('n_gen', 350),
        'problem': get_problem("dtlz2", None, 5, k=10),
        **get_setup(5)

    },

    'dtlz2_8obj': {
        'termination': ('n_gen', 500),
        'problem': get_problem("dtlz2", None, 8, k=10),
        **get_setup(8)
    },

    'dtlz2_10obj': {
        'termination': ('n_gen', 750),
        'problem': get_problem("dtlz2", None, 10, k=10),
        **get_setup(10)

    },

    'dtlz2_15obj': {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dtlz2", None, 15, k=10),
        **get_setup(15)
    },

    # ==========================================
    # DTLZ3
    # ==========================================

    'dtlz3_3obj': {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dtlz3", None, 3, k=10),
        **get_setup(3)

    },

    'dtlz3_5obj': {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dtlz3", None, 5, k=10),
        **get_setup(5)

    },

    'dtlz3_8obj': {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dtlz3", None, 8, k=10),
        **get_setup(8)
    },

    'dtlz3_10obj': {
        'termination': ('n_gen', 1500),
        'problem': get_problem("dtlz3", None, 10, k=10),
        **get_setup(10)

    },

    'dtlz3_15obj': {
        'termination': ('n_gen', 2000),
        'problem': get_problem("dtlz3", None, 15, k=10),
        **get_setup(15)
    },

    # ==========================================
    # DTLZ4
    # ==========================================

    'dtlz4_3obj': {
        'termination': ('n_gen', 600),
        'problem': get_problem("dtlz3", None, 3, k=10),
        **get_setup(3)

    },

    'dtlz4_5obj': {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dtlz4", None, 5, k=10),
        **get_setup(5)

    },

    'dtlz4_8obj': {
        'termination': ('n_gen', 1250),
        'problem': get_problem("dtlz4", None, 8, k=10),
        **get_setup(8)
    },

    'dtlz4_10obj': {
        'termination': ('n_gen', 2000),
        'problem': get_problem("dtlz4", None, 10, k=10),
        **get_setup(10)

    },

    'dtlz4_15obj': {
        'termination': ('n_gen', 3000),
        'problem': get_problem("dtlz4", None, 15, k=10),
        **get_setup(15)
    },

}

if __name__ == '__main__':

    method = "nsga3"
    n_runs = 100
    #problems = ['dtlz1/3obj']
    problems = setup.keys()

    for e in problems:

        s = setup[e]
        problem = s['problem']

        for run in range(n_runs):
            data = {
                'args': [problem, method],
                'kwargs': {
                    'method_args': {
                        'pop_size': s['pop_size'],
                        'ref_dirs': s['ref_dirs'],
                        'crossover': s['crossover'],
                        'mutation': s['mutation'],
                    },
                    'termination': s['termination']
                },
                'out': "%s/pynsga3%s_%s.out" % (e.replace("_", "/"), e, (run + 1)),
            }

            folder = "runs/pynsga3-pbi"
            os.makedirs(folder, exist_ok=True)
            fname = "pynsga3_%s_%s.run" % (e, (run + 1))
            with open(os.path.join(folder, fname), 'wb') as f:
                pickle.dump(data, f)
