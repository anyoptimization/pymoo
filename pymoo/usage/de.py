from pymoo.optimize import minimize
from pymop.factory import get_problem
from pymoo.factory import get_algorithm

problem = get_problem("rastrigin", n_var=3)

res = minimize(problem,
               method=get_algorithm('de',
                                    variant="DE/rand/1/bin",
                                    pop_size=100,
                                    CR=0.7,
                                    F=2),
               termination=('n_gen', 200),
               seed=1,
               disp=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
