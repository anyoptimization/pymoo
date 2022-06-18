import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output


class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.columns["x-mean"] = Column(13)
        self.columns["x-std"] = Column(13)
        self.active += ["x-mean", "x-std"]

    def update(self, algorithm):
        super().update(algorithm)

        X = algorithm.pop.get("X")
        self.columns["x-mean"].value = np.mean(X)
        self.columns["x-std"].value = np.std(X)


problem = get_problem("zdt2")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 10),
               seed=1,
               output=MyOutput(),
               verbose=True)
