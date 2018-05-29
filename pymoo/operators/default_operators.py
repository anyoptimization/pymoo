from pymoo.operators.crossover.bin_uniform_crossover import BinaryUniformCrossover
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.bin_bitflip_mutation import BinaryBitflipMutation
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.bin_random_sampling import BinaryRandomSampling
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.random_selection import RandomSelection


def set_if_none(kwargs, str, val):
    if str not in kwargs:
        kwargs[str] = val


def set_default_if_none(var_type, kwargs):
    set_if_none(kwargs, 'pop_size', 100)
    set_if_none(kwargs, 'verbose', False)
    set_if_none(kwargs, 'selection', RandomSelection())

    # values for mating
    if var_type == "real":
        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover())
        set_if_none(kwargs, 'mutation', PolynomialMutation())
    elif var_type == "binary":
        set_if_none(kwargs, 'sampling', BinaryRandomSampling())
        set_if_none(kwargs, 'crossover', BinaryUniformCrossover())
        set_if_none(kwargs, 'mutation', BinaryBitflipMutation())
        set_if_none(kwargs, 'eliminate_duplicates', True)
