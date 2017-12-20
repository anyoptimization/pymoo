import numpy as np

from pymoo.model import random
from pymoo.model.problem import Problem


class Knapsack(Problem):
    def __init__(self,
                 n_items,  # number of items that can be picked up
                 W,  # weights for each item
                 P,  # profit of each item
                 C,  # maximum capacity
                 ):
        Problem.__init__(self)
        self.n_var = n_items
        self.n_constr = 1
        self.n_obj = 1
        self.func = self.evaluate_

        self.W = W
        self.P = P
        self.C = C

    def evaluate_(self, x, f, g):
        g[:,0] = np.sum(self.W * x, axis=1) - self.C
        f[:, 0] = -np.sum(self.P * x, axis=1)




def create_random_knapsack_problem(n_items):
    P = random.randint(1, 100, size=n_items)
    W = random.randint(1, 100, size=n_items)
    C = np.sum(W) / 2
    problem = Knapsack(n_items, W, P, C)
    return problem