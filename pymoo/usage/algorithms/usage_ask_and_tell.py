from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.factory import Problem
from pymoo.model.evaluator import Evaluator, set_cv

from pymoo.util.termination.no_termination import NoTermination


class MyProblem(Problem):

    def __init__(self, n_var=10):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-0, xu=1)


problem = MyProblem()

algorithm = CMAES().setup(problem, termination=NoTermination(), verbose=False)

for k in range(200):
    pop = algorithm.infill()

    if pop is None:
        break

    X = pop.get("X")
    Evaluator().eval(problem, pop)

    for individual in pop:
        F = ((individual.X - 0.5) ** 2).sum()
        individual.set_by_dict(F=[F], G=0.0)
    set_cv(pop)

    algorithm.advance(infills=pop)

    print(k + 1, algorithm.opt[0].F[0])

print(algorithm.opt.get("F"))
