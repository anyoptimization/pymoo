from pymoo.optimize import minimize
from pymop.factory import get_problem

problem = get_problem("rastrigin")

res = minimize(problem,
               method='ga',
               method_args={
                   'pop_size': 100,
                   'eliminate_duplicates': False,
               },
               termination=('n_gen', 50),
               disp=True)

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)

