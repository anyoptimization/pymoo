from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100)

# prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
algorithm.setup(problem, ('n_gen', 10), seed=1, verbose=False)

# until the algorithm has no terminated
while algorithm.has_next():
    # do the next iteration
    algorithm.next()

    # do same more things, printing, logging, storing or even modifying the algorithm object
    print(algorithm.n_gen, algorithm.evaluator.n_eval)

# obtain the result objective from the algorithm
res = algorithm.result()

# calculate a hash to show that all executions end with the same result
print("hash", res.F.sum())