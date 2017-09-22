
import numpy as np

from model.evaluator import Evaluator


class Algorithm:

    def __init__(self):
        pass

    def solve(self, problem, n_eval, seed=1234):

        # set the random seed
        np.random.seed(seed)

        # call the algorithm to solve the problem
        return self.solve_(problem, Evaluator(n_eval))


