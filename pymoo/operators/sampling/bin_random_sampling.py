from pymoo.model import random


class BinaryRandomSampling:
    def sample(self, p, n):
        return random.randint(0, 1, size=(n, p.n_var))
