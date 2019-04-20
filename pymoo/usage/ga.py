from pymoo.algorithms.so_genetic_algorithm import ga
from pymoo.factory import get_crossover
from pymoo.optimize import minimize
from pymop.factory import get_problem

problem = get_problem("g01")

method = ga(pop_size=100,
            crossover=get_crossover("real_two_point_crossover"),
            eliminate_duplicates=False)

res = minimize(problem,
               method,
               termination=('n_gen', 50),
               disp=True)

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
