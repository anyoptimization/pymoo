import random

import numpy as np

from model.evaluator import Evaluator
from rand.default_random_generator import DefaultRandomGenerator


class Algorithm:

    def __init__(self):
        pass

    def solve(self, problem, evaluator, seed=1, rnd=DefaultRandomGenerator()):

        # set the random seed
        rnd.seed(seed)

        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(evaluator)

        # call the algorithm to solve the problem
        return self.solve_(problem, evaluator, rnd=rnd)


