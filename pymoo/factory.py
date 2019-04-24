"""
This module is a factory method what allows to import various objects, such as algorithms, crossover, mutation.

The definitions for each object are purposely defined as a list and not as a dictionary to keep an order for the documentation.


"""
import re

import numpy as np

from pymoo.algorithms.moead import moead
from pymoo.algorithms.nsga2 import nsga2
from pymoo.algorithms.nsga3 import nsga3
from pymoo.algorithms.rnsga2 import rnsga2
from pymoo.algorithms.rnsga3 import rnsga3
from pymoo.algorithms.so_de import de
from pymoo.algorithms.so_genetic_algorithm import ga
from pymoo.algorithms.unsga3 import unsga3
from pymoo.docs import parse_doc_string
from pymoo.model.termination import MaximumFunctionCallTermination, MaximumGenerationTermination, IGDTermination
from pymoo.operators.crossover.differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.crossover.exponential_crossover import ExponentialCrossover
from pymoo.operators.crossover.half_uniform_crossover import HalfUniformCrossover
from pymoo.operators.crossover.point_crossover import PointCrossover
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.crossover.uniform_crossover import UniformCrossover
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.operators.selection.tournament_selection import TournamentSelection


# =========================================================================================================
# Generic
# =========================================================================================================


def get_from_list(l, name, args, kwargs):
    i = None
    for k, e in enumerate(l):

        if re.match(e[0], name):
            i = k
            break

    if i is not None:

        if len(l[i]) == 2:
            name, clazz = l[i]

        elif len(l[i]) == 3:
            name, clazz, default_kwargs = l[i]

            # overwrite the default if provided
            for key, val in kwargs.items():
                default_kwargs[key] = val
            kwargs = default_kwargs

        return clazz(*args, **kwargs)
    else:
        raise Exception("Object '%s' for not found in %s" % (name, [e[0] for e in l]))


def get_options(l):
    return ", ".join(["'%s'" % k[0] for k in l])


def dummy(name, kwargs):
    """
    A convenience method to get an {type} object just by providing a string.

    Parameters
    ----------

    name : {{ {options} }}
        Name of the {type}.

    kwargs : dict
        Dictionary that should be used to call the method mapped to the {type} factory function.

    Returns
    -------
    algorithm : {clazz}
        An {type} object based on the string. `None` if the {type} was not found.

    """
    pass


# =========================================================================================================
# Algorithms
# =========================================================================================================

ALGORITHMS = [
    ("ga", ga),
    ("de", de),
    ("nsga2", nsga2),
    ("rnsga2", rnsga2),
    ("nsga3", nsga3),
    ("unsga3", unsga3),
    ("rnsga3", rnsga3),
    ("moead", moead),
]


def get_algorithm(name, *args, d={}, **kwargs):
    return get_from_list(ALGORITHMS, name, args, {**d, **kwargs})


parse_doc_string(dummy, get_algorithm, {"type": "algorithm",
                                        "clazz": ":class:`~pymoo.model.algorithm.Algorithm`",
                                        "options": get_options(ALGORITHMS)
                                        })

# =========================================================================================================
# Sampling
# =========================================================================================================

SAMPLING = [
    ("real_random", RandomSampling, {'var_type': np.real}),
    ("real_lhs", LatinHypercubeSampling),
    ("bin_random", RandomSampling, {'var_type': np.bool}),
    ("int_random", RandomSampling, {'var_type': np.int})
]


def get_sampling(name, *args, d={}, **kwargs):
    return get_from_list(SAMPLING, name, args, {**d, **kwargs})


parse_doc_string(dummy, get_sampling, {"type": "sampling",
                                       "clazz": ":class:`~pymoo.model.sampling.Sampling`",
                                       "options": get_options(SAMPLING)
                                       })

# =========================================================================================================
# Selection
# =========================================================================================================

SELECTION = [
    ("random", RandomSelection),
    ("tournament", TournamentSelection)
]


def get_selection(name, *args, d={}, **kwargs):
    return get_from_list(SELECTION, name, args, {**d, **kwargs})


parse_doc_string(dummy, get_selection, {"type": "selection",
                                        "clazz": ":class:`~pymoo.model.selection.Selection`",
                                        "options": get_options(SELECTION)
                                        })

# =========================================================================================================
# Crossover
# =========================================================================================================

CROSSOVER = [
    ("real_sbx", SimulatedBinaryCrossover, dict(prob=1.0, eta=30, var_type=np.double)),
    ("int_sbx", SimulatedBinaryCrossover, dict(prob=1.0, eta=30, var_type=np.int)),
    ("real_de", DifferentialEvolutionCrossover),
    ("(real|bin|int)_ux", UniformCrossover),
    ("(bin|int)_hux", HalfUniformCrossover),
    ("(real|bin|int)_exp", ExponentialCrossover),
    ("(real|bin|int)_one_point", PointCrossover, {'n_points': 1}),
    ("(real|bin|int)_two_point", PointCrossover, {'n_points': 2}),
    ("(real|bin|int)_k_point", PointCrossover)
]


def get_crossover(name, *args, d={}, **kwargs):
    return get_from_list(CROSSOVER, name, args, {**d, **kwargs})


parse_doc_string(dummy, get_crossover, {"type": "crossover",
                                        "clazz": ":class:`~pymoo.model.crossover.Crossover`",
                                        "options": get_options(CROSSOVER)
                                        })

# =========================================================================================================
# Mutation
# =========================================================================================================

MUTATION = [
    ("real_pm", PolynomialMutation),
    ("int_pm", PolynomialMutation, dict(var_type=np.int)),
    ("bin_bitflip", BinaryBitflipMutation)
]


def get_mutation(name, *args, d={}, **kwargs):
    return get_from_list(MUTATION, name, args, {**d, **kwargs})


parse_doc_string(dummy, get_mutation, {"type": "mutation",
                                       "clazz": ":class:`~pymoo.model.mutation.Mutation`",
                                       "options": get_options(MUTATION)
                                       })

# =========================================================================================================
# Termination
# =========================================================================================================


TERMINATION = [
    ("n_eval", MaximumFunctionCallTermination),
    ("n_gen", MaximumGenerationTermination),
    ("igd", IGDTermination)
]


def get_termination(name, *args, d={}, **kwargs):
    return get_from_list(TERMINATION, name, args, {**d, **kwargs})


parse_doc_string(dummy, get_termination, {"type": "termination",
                                          "clazz": ":class:`~pymoo.model.termination.Termination`",
                                          "options": get_options(TERMINATION)
                                          })
