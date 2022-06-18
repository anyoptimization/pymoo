from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100)

algorithm.setup(problem, termination=get_termination('n_gen', 200), verbose=True)

while algorithm.has_next():
    algorithm.next()

res = algorithm.result()

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.opt.get("F"), color="red")
plot.show()
