from pymoo.optimize import minimize
from pymop.factory import get_problem

problem = get_problem("rastrigin")

res = minimize(problem,
               method='de',
               method_args={
                   'pop_size': 100
               },
               termination=('n_gen', 200),
               disp=True)

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)

