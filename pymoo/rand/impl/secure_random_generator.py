
import random
import numpy as np

class SecureRandomGenerator:

    def random(self, size=None):
        if size is None:
            return random.SystemRandom().random()
        else:
            n = size[0]
            m = size[1]
            val = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    val[i, j] = random.SystemRandom().random()

            return val


    def permutation(self, n):
        perm = list(range(n))
        random.SystemRandom().shuffle(perm)
        return perm

    def seed(self, x):
        pass
