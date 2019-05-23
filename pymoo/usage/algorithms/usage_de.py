from pymoo.algorithms.so_de import de
from pymoo.factory import get_problem
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.optimize import minimize

method = de(
    pop_size=100,
    sampling=LatinHypercubeSampling(iterations=100, criterion="maxmin"),
    variant="DE/rand/1/bin",
    CR=0.5,
    F=0.3,
    dither="vector",
    jitter=False
)

res = minimize(get_problem("ackley", n_var=10),
               method,
               termination=('n_gen', 250),
               seed=1)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))