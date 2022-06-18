import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output


class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.x_mean = Column("x_mean", width=13, func=lambda algorithm: np.mean(algorithm.pop.get("X")))
        self.x_std = Column("x_std", width=13, func=lambda algorithm: np.std(algorithm.pop.get("X")))
        self.columns += [self.x_mean, self.x_std]

problem = get_problem("zdt2")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 10),
               seed=1,
               output=MyOutput(),
               verbose=True)


