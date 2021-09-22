import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.constraints.tcv import TotalConstraintViolation
from pymoo.core.problem import Problem
from pymoo.util.termination.no_termination import NoTermination

# set the meta-data of the problem (necessary to initialize the algorithm)
problem = Problem(n_var=30, n_obj=2, n_ieq_constr=0, xl=np.zeros(30), xu=np.ones(30))

# create the algorithm object
algorithm = NSGA2(pop_size=100)

# let the algorithm object never terminate and let the loop control it
termination = NoTermination()

# create an algorithm object that never terminates
algorithm.setup(problem, termination=termination)

# fix the random seed manually
np.random.seed(1)

# until the algorithm has no terminated
for n_gen in range(10):
    # ask the algorithm for the next solution to be evaluated
    pop = algorithm.ask()

    # get the design space values of the algorithm
    X = pop.get("X")

    # implement your evaluation. here ZDT1
    f1 = X[:, 0]
    v = 1 + 9.0 / (problem.n_var - 1) * np.sum(X[:, 1:], axis=1)
    f2 = v * (1 - np.power((f1 / v), 0.5))

    # objectives
    pop.set("F", np.column_stack([f1, f2]))

    # for constraints
    # pop.set("G", the_constraint_values))

    # set the total constraint violation
    TotalConstraintViolation().do(pop)

    # returned the evaluated individuals which have been evaluated or even modified
    algorithm.tell(infills=pop)

    # do same more things, printing, logging, storing or even modifying the algorithm object
    print(algorithm.n_gen)

# obtain the result objective from the algorithm
res = algorithm.result()

# calculate a hash to show that all executions end with the same result
print("hash", res.F.sum())