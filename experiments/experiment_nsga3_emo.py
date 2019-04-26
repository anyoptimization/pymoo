import os
import pickle

from experiments.experiment_nsga3_old import get_setup
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.experimental.emo.max_non_dominated import ReferenceDirectionSurvivalNonDominated
from pymoo.experimental.emo.max_of_extremes import ReferenceDirectionSurvivalMaxExtremes
from pymoo.experimental.emo.true import ReferenceDirectionSurvivalTrue
from pymoo.model.termination import MaximumGenerationTermination
from pymop import ScaledProblem
from pymop.factory import get_problem

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

    'dtlz1_10obj': {
        'termination': ('n_gen', 1000),
        'problem': get_problem("dtlz1", None, 10, k=5),
        **get_setup(10)
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

    'dtlz2_10obj': {
        'termination': ('n_gen', 750),
        'problem': get_problem("dtlz2", None, 10, k=10),
        **get_setup(10)

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

    'dtlz3_10obj': {
        'termination': ('n_gen', 1500),
        'problem': get_problem("dtlz3", None, 10, k=10),
        **get_setup(10)

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

    'dtlz4_10obj': {
        'termination': ('n_gen', 2000),
        'problem': get_problem("dtlz4", None, 10, k=10),
        **get_setup(10)

    },

    # ==========================================
    # Scaled DTLZ1
    # ==========================================

    'sdtlz1_3obj': {
        'termination': ('n_gen', 400),
        'problem': ScaledProblem(get_problem("dtlz1", None, 3, k=5), 10),
        **get_setup(3)

    },

    'sdtlz1_5obj': {
        'termination': ('n_gen', 600),
        'problem': ScaledProblem(get_problem("dtlz1", None, 5, k=5), 10),
        **get_setup(5)
    },

    'sdtlz1_10obj': {
        'termination': ('n_gen', 1000),
        'problem': ScaledProblem(get_problem("dtlz1", None, 10, k=5), 2),
        **get_setup(10)
    },

    # ==========================================
    # Scaled DTLZ2
    # ==========================================

    'sdtlz2_3obj': {
        'termination': ('n_gen', 250),
        'problem': ScaledProblem(get_problem("dtlz2", None, 3, k=10), 10),
        **get_setup(3)

    },

    'sdtlz2_5obj': {
        'termination': ('n_gen', 350),
        'problem': ScaledProblem(get_problem("dtlz2", None, 5, k=10), 10),
        **get_setup(5)

    },

    'sdtlz2_10obj': {
        'termination': ('n_gen', 750),
        'problem': ScaledProblem(get_problem("dtlz2", None, 10, k=10), 3),
        **get_setup(10)

    },

}

if __name__ == '__main__':

    run_files = []
    prefix = "runs"
    method_name = "pynsga3-true"
    n_runs = 50
    problems = setup.keys()

    for e in problems:

        s = setup[e]
        problem = s['problem']

        s = setup[e]
        problem = s['problem']
        pf = problem.pareto_front(s['ref_dirs'])

        algorithm = NSGA3(s['ref_dirs'],
                          pop_size=s['pop_size'],
                          crossover=s['crossover'],
                          mutation=s['mutation'],
                          survival=ReferenceDirectionSurvivalTrue(s['ref_dirs'], pf)
                          )

        for run in range(n_runs):
            fname = "%s_%s_%s.run" % (method_name, e, (run + 1))

            data = {
                'problem': problem,
                'algorithm': algorithm,
                'seed': run,
                'termination': MaximumGenerationTermination(s['termination'][1]),
                'out': "results/%s/%s/%s_%s.out" % (method_name, e.replace("_", "/"), e, (run + 1)),
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
