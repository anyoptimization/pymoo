from pyDOE import lhs
import diversipy
from rand.default_random_generator import DefaultRandomGenerator


class LHS:
    def sample(self, n, xl, xu, rnd=DefaultRandomGenerator(), impl='diversipy'):
        m = len(xl)

        if impl == 'diversipy':
            val = diversipy.hycusampling.maximin_reconstruction(n, len(xl))
        elif impl =='pyDOE':
            val = lhs(m, samples=n)
        else:
            return None

        for i in range(m):
            val[:, i] = val[:, i] * (xu[i] - xl[i]) + xl[i]
        return val


