import os
import pickle

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.util.reference_direction import MultiLayerReferenceDirectionFactory, UniformReferenceDirectionFactory
from pymop.factory import get_problem


def get_setup(n_obj):
    if n_obj == 2:
        pop_size = 100
        ref_dirs = UniformReferenceDirectionFactory(n_obj, pop_size - 1).do()
    elif n_obj == 3:
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
    # MISC
    # ==========================================

    # ==========================================
    # C1-DTLZ1
    # ==========================================

    'c1-dtlz1-3obj': {
        'termination': ('n_gen', 500),
        'problem': get_problem("c1dtlz1", n_obj=3),
        **get_setup(3),
    },

    'c1-dtlz1-5obj': {
        'termination': ('n_gen', 600),
        'problem': get_problem("c1dtlz1", n_obj=5),
        **get_setup(5),
    },


    # ==========================================
    # C1-DTLZ3
    # ==========================================

    'c1-dtlz3-3obj': {
        'termination': ('n_gen', 1000),
        'problem': get_problem("c1dtlz3", n_obj=3),
        **get_setup(3),
    },

    'c1-dtlz3-5obj': {
        'termination': ('n_gen', 1500),
        'problem': get_problem("c1dtlz3", n_obj=5),
        **get_setup(5),
    },


    # ==========================================
    # C2-DTLZ2
    # ==========================================

    'c2-dtlz2-3obj': {
        'termination': ('n_gen', 250),
        'problem': get_problem("c2dtlz2", n_obj=3),
        **get_setup(3),
    },

    'c2-dtlz2-5obj': {
        'termination': ('n_gen', 350),
        'problem': get_problem("c2dtlz2", n_obj=5),
        **get_setup(5),
    },



    # ==========================================
    # C3-DTLZ4
    # ==========================================

    'c3-dtlz4-3obj': {
        'termination': ('n_gen', 750),
        'problem': get_problem("c3dtlz4", n_obj=3),
        **get_setup(3),
    },

    'c3-dtlz4-5obj': {
        'termination': ('n_gen', 1250),
        'problem': get_problem("c3dtlz4", n_obj=5),
        **get_setup(5),
    },





}


if __name__ == '__main__':

    run_files = []
    prefix = "runs"
    method_name = "pynsga3-parameter-less"
    n_runs = 31

    for key in setup.keys():

        s = setup[key]
        problem = s['problem']

        for run in range(n_runs):
            fname = "%s_%s_%s.run" % (method_name, key, (run + 1))

            data = {
                'args': [problem, "nsga3"],
                'kwargs': {
                    'method_args': {
                        'pop_size': s['pop_size'],
                        'ref_dirs': s['ref_dirs'],
                        'crossover': s['crossover'],
                        'mutation': s['mutation'],
                    },
                    'termination': s['termination'],
                    'seed': (run + 1)
                },
                'out': "%s/%s/%s_%s_%s.out" % (method_name, key, method_name, key, (run + 1)),
                'in': os.path.join(prefix, method_name, fname),
            }

            os.makedirs(os.path.join(prefix, method_name), exist_ok=True)

            with open(data['in'], 'wb') as f:
                pickle.dump(data, f)
                run_files.append(data)

        # create the final run.txt file
        with open(os.path.join(prefix, method_name, "run.bat"), 'w') as f:
            for run_file in run_files:
                f.write("python execute.py %s %s\n" % (run_file['in'], run_file['out']))
