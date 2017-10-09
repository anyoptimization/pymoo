
from rand.default_random_generator import DefaultRandomGenerator


class RandomFactory:
    def sample(self, n, xl, xu, rnd=DefaultRandomGenerator()):
        m = len(xl)
        val = rnd.random(size=(n, m))
        for i in range(m):
            val[:, i] = val[:, i] * (xu[i] - xl[i]) + xl[i]
        return val


