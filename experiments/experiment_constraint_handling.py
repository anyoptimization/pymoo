import os
import pickle

from pymop.factory import get_problem

setup = {

    'G1': {
        'termination': ('n_gen', 1750),
        'problem': "g01",
    },

    'G2': {
        'termination': ('n_gen', 1750),
        'problem': "g02",
    },

    'G3': {
        'termination': ('n_gen', 1750),
        'problem': "g03",
    },

    'G4': {
        'termination': ('n_gen', 1750),
        'problem': "g04",
    },

    'G5': {
        'termination': ('n_gen', 1750),
        'problem': "g05",
    },

    'G6': {
        'termination': ('n_gen', 1750),
        'problem': "g06",
    },

    'G7': {
        'termination': ('n_gen', 1750),
        'problem': "g07",
    },

    'G8': {
        'termination': ('n_gen', 1750),
        'problem': "g08",
    },

    'G9': {
        'termination': ('n_gen', 1750),
        'problem': "g09",
    },

    'G10': {
        'termination': ('n_gen', 1750),
        'problem': "g10",
    }

}

if __name__ == '__main__':

    run_files = []
    prefix = "runs"
    method_name = "pyde-rand+best-exp"
    n_runs = 31
    entries = setup.keys()

    for key in entries:

        s = setup[key]
        str_problem = s['problem']
        problem = get_problem(str_problem)

        for run in range(n_runs):
            fname = "%s_%s_%s.run" % (method_name, str_problem, (run + 1))

            data = {
                'args': [problem, "de"],
                'kwargs': {
                    'method_args': {
                        'pop_size': 200
                    },
                    'termination': s['termination'],
                    'seed': (run + 1)
                },
                'out': "%s/%s/%s_%s_%s.out" % (method_name, str_problem, method_name, str_problem, (run + 1)),
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
