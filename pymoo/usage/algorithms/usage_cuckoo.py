from pymoo.algorithms.so_cuckoo_search import CuckooSearch
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("ackley", n_var=10)

algorithm = CuckooSearch(pop_size=50, alpha=0.05, pa=0.1)

res = minimize(problem,
               algorithm,
               # callback=TwoVariablesOneObjectiveVisualization(do_show=True, do_close=True),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))