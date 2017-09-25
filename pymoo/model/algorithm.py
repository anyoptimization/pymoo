import random

import numpy as np

from model.evaluator import Evaluator


class Algorithm:

    def __init__(self):
        pass

    def solve(self, problem, evaluator, seed=1234):

        # set the random seed
        random.seed(seed)
        np.random.seed(seed)

        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(evaluator)

        # call the algorithm to solve the problem
        return self.solve_(problem, evaluator)


