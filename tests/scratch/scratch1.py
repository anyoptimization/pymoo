from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem, get_termination
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.visualization.scatter import Scatter

problem = get_problem("rastrigin")

termination = SingleObjectiveDefaultTermination(x_tol=1e-100,
                                                cv_tol=1e-6,
                                                f_tol=1e-6,
                                                nth_gen=5,
                                                n_last=20,
                                                n_max_gen=5)

algorithm = GA(pop_size=100, eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
