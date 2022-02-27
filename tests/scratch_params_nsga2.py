from copy import deepcopy

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.recorder import DefaultSingleObjectiveRecorder, DefaultMultiObjectiveRecorder
from pymoo.factory import get_problem
from pymoo.operators.param_control import RandomParameterControl, EvolutionaryParameterControl, NoParameterControl
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

# problem = get_problem("wfg3", n_var=30, n_obj=2)

n_gen = 400

seed = 2

save_history = False

if problem.n_obj == 1:
    recorder = DefaultSingleObjectiveRecorder()
    indicator = "fgap"
else:
    recorder = DefaultMultiObjectiveRecorder()
    indicator = "igd"

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))

algorithm = NSGA2(control=NoParameterControl)

res_a = minimize(problem,
                 algorithm,
                 seed=seed,
                 callback=deepcopy(recorder),
                 termination=('n_gen', n_gen),
                 save_history=save_history,
                 verbose=True)

n_evals, perf = res_a.algorithm.callback.get("n_evals", indicator)
plt.plot(n_evals, perf, color='black', lw=0.7, label="Vanilla")

algorithm = NSGA2(control=RandomParameterControl)

res_b = minimize(problem,
                 algorithm,
                 seed=seed,
                 callback=deepcopy(recorder),
                 termination=('n_gen', n_gen),
                 save_history=save_history,
                 verbose=True)
n_evals, perf = res_b.algorithm.callback.get("n_evals", indicator)
plt.plot(n_evals, perf, color='red', lw=0.7, label="RandomParameterControl")

algorithm = NSGA2(control=EvolutionaryParameterControl)

res_c = minimize(problem,
                 algorithm,
                 seed=seed,
                 callback=deepcopy(recorder),
                 termination=('n_gen', n_gen),
                 save_history=save_history,
                 verbose=True)

n_evals, perf = res_c.algorithm.callback.get("n_evals", indicator)
plt.plot(n_evals, perf, color='blue', lw=0.7, label="EvolutionaryParameterControl")

plt.yscale("log")
plt.legend()
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("IGD")
plt.show()

sc = Scatter()
sc.add(res_a.F, label="Vanilla", color="black")
sc.add(res_b.F, label="Random", color="red")
sc.add(res_c.F, label="Genetic", color="red")
sc.add(problem.pareto_front(), plot_type="line")
sc.show()
