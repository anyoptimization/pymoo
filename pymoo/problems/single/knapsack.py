import autograd.numpy as anp

from pymoo.model.problem import Problem


class Knapsack(Problem):
    def __init__(self,
                 n_items,  # number of items that can be picked up
                 W,  # weights for each item
                 P,  # profit of each item
                 C,  # maximum capacity
                 ):
        super().__init__(n_var=n_items, n_obj=1, n_constr=1, xl=0, xu=1, type_var=anp.bool)

        self.W = W
        self.P = P
        self.C = C

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -anp.sum(self.P * x, axis=1)
        out["G"] = (anp.sum(self.W * x, axis=1) - self.C)


def create_random_knapsack_problem(n_items, seed=1):
    anp.random.seed(seed)
    P = anp.random.randint(1, 100, size=n_items)
    W = anp.random.randint(1, 100, size=n_items)
    C = int(anp.sum(W) / 10)
    problem = Knapsack(n_items, W, P, C)
    return problem
