from pymoo.algorithms.so_de import DE
from pymoo.factory import get_problem
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.optimize import minimize


problem = get_problem("ackley", n_var=10)

algorithm = DE(
    pop_size=100,
    sampling=LatinHypercubeSampling(iterations=100, criterion="maxmin"),
    variant="DE/rand/1/bin",
    CR=0.5,
    F=0.3,
    dither="vector",
    jitter=False
)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))