from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


def my_callback(algorithm):
    disp = algorithm.func_display_attrs(algorithm.problem, algorithm.evaluator, algorithm, algorithm.pf)
    print("My Custom Output: ", end='')
    algorithm._display(disp)


problem = get_problem("zdt2")

algorithm = NSGA2(pop_size=100, elimate_duplicates=True, callback=my_callback)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
