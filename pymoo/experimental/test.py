from pymoo.algorithms.so_genetic_algorithm import ga
from pymoo.optimize import minimize
from pymop.factory import get_problem

method = ga(
    pop_size=100,
    eliminate_duplicates=True)

res = minimize(get_problem("g01"),
               method,
               termination=('n_gen', 50),
               disp=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
