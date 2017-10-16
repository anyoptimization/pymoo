from pyDOE import lhs

from rand.default_random_generator import DefaultRandomGenerator


class LHS:
    def sample(self, n, xl, xu, rnd=DefaultRandomGenerator()):
        m = len(xl)
        val = lhs(m, samples=n)
        for i in range(m):
            val[:, i] = val[:, i] * (xu[i] - xl[i]) + xl[i]
        return val


