from pymoo.rand.impl.numpy_random_generator import NumpyRandomGenerator
import os


class Configuration:
    rand = NumpyRandomGenerator()


# returns the directory to be used for imports
def get_pymoo():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
