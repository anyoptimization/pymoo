import numpy as np

from pymoo.problems import get_problem
from pymoo.visualization.util import plot, plot_problem_surface
from pymoo.visualization.scatter import Scatter

# --------------------------------------------------------------------------------------------
# Single
# --------------------------------------------------------------------------------------------

problem = get_problem("ackley", n_var=2, a=20, b=1 / 5, c=2 * np.pi)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")

problem = get_problem("rastrigin", n_var=2)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")

problem = get_problem("sphere", n_var=2)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")

problem = get_problem("zakharov", n_var=2)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")

# --------------------------------------------------------------------------------------------
# Multi
# --------------------------------------------------------------------------------------------


problem = get_problem("zdt1")
plot(problem.pareto_front(), no_fill=True)

problem = get_problem("zdt2")
plot(problem.pareto_front(), no_fill=True)

problem = get_problem("zdt3")
plot(problem.pareto_front(), no_fill=True)

problem = get_problem("zdt4")
plot(problem.pareto_front(), no_fill=True)

problem = get_problem("zdt5", normalize=False)
plot(problem.pareto_front(), no_fill=True)

problem = get_problem("zdt5")
plot(problem.pareto_front(), no_fill=True)

problem = get_problem("zdt6")
plot(problem.pareto_front(), no_fill=True)

problem = get_problem("bnh")
plot(problem.pareto_front(), no_fill=True)

problem = get_problem("rosenbrock", n_var=2)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")

problem = get_problem("griewank", n_var=2)
plot_problem_surface(problem, 100, plot_type="wireframe+contour")

pf = get_problem("truss2d").pareto_front()

plot = Scatter(title="Pareto-front")
plot.add(pf, s=80, facecolors='none', edgecolors='r')
plot.add(pf, plot_type="line", color="black", linewidth=2)
plot.show()

plot.reset()
plot.do()
plot.apply(lambda ax: ax.set_yscale("log"))
plot.apply(lambda ax: ax.set_xscale("log"))
plot.show()

