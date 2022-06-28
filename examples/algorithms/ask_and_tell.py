from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

problem = Problem(n_var=10, n_obj=1, n_ieq_constr=1, xl=-0, xu=1)

algorithm = PSO().setup(problem, termination=NoTermination(), verbose=False)

for k in range(20):

    if not algorithm.has_next():
        break

    infills = algorithm.ask()

    X = infills.get("X")

    F = (X ** 2).sum(axis=1)
    G = - (X[:, 0] + X[:, 1]) - 0.3

    algorithm.evaluator.eval(StaticProblem(problem, F=F, G=G), infills)

    algorithm.tell(infills=infills)

    print(k + 1, algorithm.opt[0].F[0])

print(algorithm.opt.get("F"))
