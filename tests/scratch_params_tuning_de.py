

import json
import pickle
import sys

from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))

from pymoo.algorithms.param_tuning import ParameterTuning
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.core.parameters import get_params
from pymoo.factory import get_problem

instance = int(sys.argv[1])


n_jobs = 1

n_trails = 300

n_runs = 11

pt = ParameterTuning(n_trails, runs=n_runs)

algorithm = G3PCX()

params = get_params(algorithm)
print(params)

problem = get_problem(f"bbob-f{instance}-1", n_var=10)

ret = pt.do(problem, algorithm, termination=("n_evals", 20_000), n_jobs=n_jobs)

print("DONE")
print(ret["params"], ret["performance"])

fname = f"bbob-f{instance}.json"
with open(fname, 'w') as f:
    json.dump(ret["params"], f)

fname = f"bbob-f{instance}.pkl"
with open(fname, 'wb') as f:
    pickle.dump(ret, f)





