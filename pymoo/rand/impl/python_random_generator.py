import random

from pymoo.rand.random_generator import RandomGenerator


class PythonRandomGenerator(RandomGenerator):

    def _seed(self, n):
        random.seed(n)

    def _rand_float(self):
        return random.random()
