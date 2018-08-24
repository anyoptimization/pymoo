import random

from pymoo.rand.random_generator import RandomGenerator


class SecureRandomGenerator(RandomGenerator):

    def _seed(self, n):
        random.SystemRandom().seed(n)

    def _rand_float(self):
        return random.SystemRandom().random()




