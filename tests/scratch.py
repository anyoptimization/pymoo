from pymoo.algorithms.moo.gde3 import GDE3
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.factory import get_problem, ZDT1, ZDT2, ZDT3, ZDT4
from pymoo.optimize import minimize


# problem = get_problem("ackley", n_var=30)

problem = ZDT4()

algorithm = GDE3(
    pop_size=100,
    dither="vector",
    jitter=True
)

res = minimize(problem,
               algorithm,
               ('n_gen', 600),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
