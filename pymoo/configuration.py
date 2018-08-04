



class Configuration:
    EPS = 1e-30

    from pymoo.rand.impl.my_random_generator import MyRandomGenerator
    rand = MyRandomGenerator()

    #from pymoo.rand.impl.numpy_random_generator import NumpyRandomGenerator
    #rand = NumpyRandomGenerator()