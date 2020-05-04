from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import ZDT1
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.util.termination.max_gen import MaximumGenerationTermination

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

problem = ZDT1()

res = minimize(problem,
               algorithm,
               ('n_gen', 40),
               seed=1,
               pf=problem.pareto_front(use_cache=False),
               save_history=True,
               verbose=True)

algorithm.sampling = res.algorithm.pop

res = minimize(problem,
               algorithm,
               ('n_gen', 20),
               seed=1,
               pf=problem.pareto_front(use_cache=False),
               save_history=True,
               verbose=True)

