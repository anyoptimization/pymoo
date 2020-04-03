from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt3")
algorithm = NSGA2(pop_size=100)
termination = MultiObjectiveSpaceToleranceTermination(tol=0.0025,
                                                      n_last=30,
                                                      nth_gen=5,
                                                      n_max_gen=None,
                                                      n_max_evals=None)

res = minimize(problem,
               algorithm,
               termination,
               pf=True,
               seed=1,
               verbose=False)

print("Generations", res.algorithm.n_gen)
plot = Scatter(title="ZDT3")
plot.add(problem.pareto_front(use_cache=False, flatten=False), plot_type="line", color="black")
plot.add(res.F, color="red", alpha=0.8, s=20)
plot.show()
