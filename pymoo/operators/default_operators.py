from pymoo.operators.crossover.uniform_crossover import UniformCrossover
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.random_selection import RandomSelection


def set_if_none(kwargs, str, val):
    if str not in kwargs:
        kwargs[str] = val


def set_default_if_none(var_type, kwargs):
    set_if_none(kwargs, 'pop_size', 100)
    set_if_none(kwargs, 'disp', False)
    set_if_none(kwargs, 'selection', RandomSelection())
    set_if_none(kwargs, 'survival', None)

    # values for mating
    if var_type == "real":
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=0.9, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=15))
    elif var_type == "binary":
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'crossover', UniformCrossover())
        set_if_none(kwargs, 'mutation', BinaryBitflipMutation())
        set_if_none(kwargs, 'eliminate_duplicates', True)
