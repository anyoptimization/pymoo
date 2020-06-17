
# --------------------------------------------------------------------------------------------
# Single
# --------------------------------------------------------------------------------------------

# START ackley
import numpy as np

from pymoo.factory import get_problem
from pymoo.util.plotting import plot_problem_surface

problem = get_problem("ackley", n_var=2, a=20, b= 1 /5, c=2 * np.pi)
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
from pymoo.model.problem import FunctionalProblem


objs = [
    lambda x: np.sum((x - 2) ** 2),
    lambda x: np.sum((x + 2) ** 2)
]

constr_ieq = [
    lambda x: np.sum((x - 1) ** 2)
]


problem = FunctionalProblem(10,
                            objs,
                            constr_ieq=constr_ieq,
                            xl=np.array([-10, -5, -10]),
                            xu=np.array([10, 5, 10])
                            )

F, CV = problem.evaluate(np.random.rand(3, 10))

print(f"F: {F}\n")
print(f"CV: {CV}")

# END from_string
