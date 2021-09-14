from pymoo.factory import MODAct
from pymoo.util.plotting import plot

problem = MODAct("cs3")
plot(problem.pareto_front(), no_fill=True)