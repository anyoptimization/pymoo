import numpy as np
import numpy as np

from pymoo.core.problem import Problem


class Knapsack(Problem):
    def __init__(self,
                 n_items,  # number of items that can be picked up
                 W,  # weights for each item
                 P,  # profit of each item
                 C,  # maximum capacity
                 ):
        super().__init__(n_var=n_items, n_obj=1, n_ieq_constr=1, xl=0, xu=1, vtype=bool)

        self.W = W
        self.P = P
        self.C = C

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -np.sum(self.P * x, axis=1)
        out["G"] = (np.sum(self.W * x, axis=1) - self.C)


class MultiObjectiveKnapsack(Knapsack):
    def __init__(self, *args):
        super().__init__(*args)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = - np.sum(self.P * x, axis=1)
        f2 = np.sum(x, axis=1)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = (np.sum(self.W * x, axis=1) - self.C)


def create_random_knapsack_problem(n_items, seed=1, variant="single"):
    np.random.seed(seed)
    P = np.random.randint(1, 100, size=n_items)
    W = np.random.randint(1, 100, size=n_items)
    C = int(np.sum(W) / 10)

    if variant == "single":
        problem = Knapsack(n_items, W, P, C)
    else:
        problem = MultiObjectiveKnapsack(n_items, W, P, C)

    return problem
