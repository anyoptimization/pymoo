from pymoo.rand.impl.numpy_random_generator import NumpyRandomGenerator
import os


class Configuration:

    # the random generator to be used - default is numpy
    rand = NumpyRandomGenerator()


# returns the directory to be used for imports
def get_pymoo():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
