from pymoo.rand.impl.custom_random_generator import CustomRandomGenerator
from pymoo.rand.impl.numpy_random_generator import NumpyRandomGenerator


class Configuration:
    EPS = 1e-30
    INF = 1e+14
    rand = NumpyRandomGenerator()


