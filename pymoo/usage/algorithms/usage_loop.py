from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem, get_termination
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100, termination=get_termination('n_gen', 200))

algorithm.setup(problem, verbose=True)

while algorithm.has_next():
    algorithm.next()

res = algorithm.result()

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.opt.get("F"), color="red")
plot.show()
