from copy import deepcopy

import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.core.parameters import get_params
from pymoo.core.recorder import DefaultSingleObjectiveRecorder
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("rosenbrock", n_var=10)
# problem = get_problem("bbob-f01-1")

termination = ("n_gen", 300)

seed = 3

verbose = True
save_history = False

recorder = DefaultSingleObjectiveRecorder()
indicator = "fgap"

algorithm = G3PCX()

params = get_params(algorithm)

res = minimize(problem,
               algorithm,
               seed=seed,
               callback=deepcopy(recorder),
               termination=termination,
               save_history=save_history,
               verbose=verbose)

n_evals, perf = res.algorithm.callback.get("n_evals", indicator)

# plt.figure(figsize=(7, 5))
# plt.plot(n_evals, perf, color='black', lw=0.7, label="Vanilla")
# plt.yscale("log")
# plt.legend()
# plt.title("Convergence")
# plt.xlabel("Function Evaluations")
# plt.ylabel("IGD")
# plt.show()
