

# --------------------------------------------------------------------------------------------
# Single
# --------------------------------------------------------------------------------------------

# START ackley
import numpy as np

from pymoo.factory import get_problem
from pymoo.util.plotting import plot_problem_surface

problem = get_problem("ackley", n_var=2, a=20, b=1/5, c=2 * np.pi)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")
# END ackley

# START rastrigin
from pymoo.factory import get_problem
from pymoo.util.plotting import plot_problem_surface

problem = get_problem("rastrigin", n_var=2)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")
# END rastrigin

# START sphere
from pymoo.factory import get_problem
from pymoo.util.plotting import plot_problem_surface

problem = get_problem("sphere", n_var=2)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")
# END sphere

# START zakharov
from pymoo.factory import get_problem
from pymoo.util.plotting import plot_problem_surface

problem = get_problem("zakharov", n_var=2)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")
# END zakharov



# --------------------------------------------------------------------------------------------
# Multi
# --------------------------------------------------------------------------------------------


# START zdt1
from pymoo.factory import get_problem
from pymoo.util.plotting import plot

problem = get_problem("zdt1")
plot(problem.pareto_front(), no_fill=True)
# END zdt1

# START zdt2
from pymoo.factory import get_problem
from pymoo.util.plotting import plot

problem = get_problem("zdt2")
plot(problem.pareto_front(), no_fill=True)
# END zdt2

# START zdt3
from pymoo.factory import get_problem
from pymoo.util.plotting import plot

problem = get_problem("zdt3")
plot(problem.pareto_front(), no_fill=True)
# END zdt3

# START zdt4
from pymoo.factory import get_problem
from pymoo.util.plotting import plot

problem = get_problem("zdt4")
plot(problem.pareto_front(), no_fill=True)
# END zdt4

# START zdt5_no_norm
from pymoo.factory import get_problem
from pymoo.util.plotting import plot

problem = get_problem("zdt5", normalize=False)
plot(problem.pareto_front(), no_fill=True)
# END zdt5_no_norm

# START zdt5
from pymoo.factory import get_problem
from pymoo.util.plotting import plot

problem = get_problem("zdt5")
plot(problem.pareto_front(), no_fill=True)
# END zdt5

# START zdt6
from pymoo.factory import get_problem
from pymoo.util.plotting import plot

problem = get_problem("zdt6")
plot(problem.pareto_front(), no_fill=True)
# END zdt6

# START bnh
from pymoo.factory import get_problem
from pymoo.util.plotting import plot

problem = get_problem("bnh")
plot(problem.pareto_front(), no_fill=True)
# END bnh


# START rosenbrock
from pymoo.factory import get_problem
from pymoo.util.plotting import plot_problem_surface

problem = get_problem("rosenbrock", n_var=2)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")
# END rosenbrock

# START griewank
from pymoo.factory import get_problem
from pymoo.util.plotting import plot_problem_surface

problem = get_problem("griewank", n_var=2)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")
# END griewank

# START truss2d
from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter

pf = get_problem("truss2d").pareto_front()

plot = Scatter(title="Pareto-front")
plot.add(pf, s=80, facecolors='none', edgecolors='r')
plot.add(pf, plot_type="line", color="black", linewidth=2)
plot.show()
# END truss2d

# START truss2d_log
plot.reset()
plot.do()
plot.apply(lambda ax: ax.set_yscale("log"))
plot.apply(lambda ax: ax.set_xscale("log"))
plot.show()
# END truss2d_log




# --------------------------------------------------------------------------------------------


# START from_string

from pymoo.factory import get_problem

p = get_problem("dtlz1_-1", n_var=20, n_obj=5)

# create a simple test problem from string
p = get_problem("Ackley")

# the input name is not case sensitive
p = get_problem("ackley")

# also input parameter can be provided directly
p = get_problem("dtlz1_-1", n_var=20, n_obj=5)

# END from_string


# --------------------------------------------------------------------------------------------

# START from_function

import numpy as np

# this will be the evaluation function that is called each time
from pymoo.model.problem import get_problem_from_func


def my_evaluate_func(x, out, *args, **kwargs):

    # define the objective as x^2
    f1 = np.sum(np.square(x - 2), axis=1)
    f2 = np.sum(np.square(x + 2), axis=1)
    out["F"] = np.column_stack([f1, f2])

    # x^2 < 2 constraint
    out["G"] = np.sum(np.square(x - 1), axis=1)


# load the problem from a function - define 3 variables with the same lower bound
problem = get_problem_from_func(my_evaluate_func, xl=-10, xu=10, n_var=3)
F, CV = problem.evaluate(np.random.rand(100, 3))

# or define a problem with varying lower and upper bounds
problem = get_problem_from_func(my_evaluate_func, xl=np.array([-10, -5, -10]), xu=np.array([10, 5, 10]))
F, CV = problem.evaluate(np.random.rand(100, 3))

# END from_string