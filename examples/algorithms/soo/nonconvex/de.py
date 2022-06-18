from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.problems import get_problem
from pymoo.optimize import minimize


problem = get_problem("ackley", n_var=20)

algorithm = DE(
    pop_size=100,
    variant="DE/rand/1/bin",
    CR=0.3,
    dither="vector",
    jitter=True
)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
