from pymoo.rand.my_random_generator import MyRandomGenerator
from pymoo.rand.numpy_random_generator import NumpyRandomGenerator


class Configuration:
    EPS = 1e-30
    BENCHMARK_DIR = '/Users/julesy/benchmark/'
    rand = MyRandomGenerator()
    #rand = NumpyRandomGenerator()

