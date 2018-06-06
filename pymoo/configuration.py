from pymoo.rand.impl.my_random_generator import MyRandomGenerator
from pymoo.rand.impl.numpy_random_generator import NumpyRandomGenerator
from pymoo.rand.impl.secure_random_generator import SecureRandomGenerator
from pymoo.rand.impl.default_random_generator import DefaultRandomGenerator

class Configuration:
    EPS = 1e-30
    rand = NumpyRandomGenerator()