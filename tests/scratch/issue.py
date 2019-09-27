import autograd.numpy as anp
import numpy as np
from pymoo.model.problem import Problem

# Define problem parameters
knobs = 2  # number of variables (decision space dimension)
number_of_objectives = 2  # number of objectives (all to be minimized)
number_of_constraints = 2  # number of constraints (all have the form const_i <= 0)
lbound = [-2, -2]  # set lower bounds for all decision variables
ubound = [+2. + 2]  # set upper bounds for all decision variables
number_of_threads = 4  # number of threads


# Define problem
class task(Problem):

    # Set main parameters
    def __init__(self, **kwargs):
        super().__init__(
            n_var=knobs,
            n_obj=number_of_objectives,
            n_constr=number_of_constraints,
            xl=lbound,
            xu=ubound,
            elementwise_evaluation=True,
            **kwargs
        )

    # Set objectives and constraints
    def _evaluate(self, x, out, *args, **kwargs):
        # Define objective functions and constraints
        f1 = x[0] ** 2 + x[1] ** 2
        f2 = (x[0] - 1) ** 2 + x[1] ** 2
        g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        g2 = -20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8
        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


# Set problem
problem = task(parallelization=("threads", number_of_threads))

#  Algorithm
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

algorithm = NSGA2(
    pop_size=50,
    n_offsprings=20,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

# Termination
from pymoo.factory import get_termination

termination = get_termination("n_gen", 30)

# Optimize (works OK without verbose)
from pymoo.optimize import minimize

res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)

# Plot
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

plot = Scatter(title="Design Space", axis_labels="x")
plot.add(res.X, s=30, facecolors='none', edgecolors='r')
plot.do()
plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
plot.apply(lambda ax: ax.set_ylim(-2, 2))
plt.show()

plot = Scatter(title="Objective Space")
plot.add(res.F)
plot.do()
plt.show()