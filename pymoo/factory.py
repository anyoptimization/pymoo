"""
This module is a factory method what allows to import various objects, such as algorithms, crossover, mutation.

The definitions for each object are purposely defined as a list and not as a dictionary to keep an order for the documentation.


"""
import re

from pymoo.configuration import Configuration
from pymoo.problems.many import *
from pymoo.problems.multi import *
from pymoo.problems.single import *


# =========================================================================================================
# Generic
# =========================================================================================================


def get_from_list(l, name, args, kwargs):
    i = None

    for k, e in enumerate(l):
        if e[0] == name:
            i = k
            break

    if i is None:
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


# =========================================================================================================
# Algorithms
# =========================================================================================================

def get_algorithm_options():
    from pymoo.algorithms.ctaea import CTAEA
    from pymoo.algorithms.moead import MOEAD
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.algorithms.nsga3 import NSGA3
    from pymoo.algorithms.rnsga2 import RNSGA2
    from pymoo.algorithms.rnsga3 import RNSGA3
    from pymoo.algorithms.so_de import DE
    from pymoo.algorithms.so_genetic_algorithm import GA
    from pymoo.algorithms.unsga3 import UNSGA3
    from pymoo.algorithms.so_nelder_mead import NelderMead
    from pymoo.algorithms.so_cmaes import CMAES
    from pymoo.algorithms.so_brkga import BRKGA
    from pymoo.algorithms.so_pattern_search import PatternSearch

    ALGORITHMS = [
        ("ga", GA),
        ("brkga", BRKGA),
        ("de", DE),
        ("nelder-mead", NelderMead),
        ("pattern-search", PatternSearch),
        ("cmaes", CMAES),
        ("nsga2", NSGA2),
        ("rnsga2", RNSGA2),
        ("nsga3", NSGA3),
        ("unsga3", UNSGA3),
        ("rnsga3", RNSGA3),
        ("moead", MOEAD),
        ("ctaea", CTAEA),
    ]

    return ALGORITHMS


def get_algorithm(name, *args, d={}, **kwargs):
    return get_from_list(get_algorithm_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Sampling
# =========================================================================================================

def get_sampling_options():
    from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
    from pymoo.operators.sampling.random_sampling import FloatRandomSampling
    from pymoo.operators.integer_from_float_operator import IntegerFromFloatSampling
    from pymoo.operators.sampling.random_sampling import BinaryRandomSampling
    from pymoo.operators.sampling.random_permutation_sampling import PermutationRandomSampling

    SAMPLING = [
        ("real_random", FloatRandomSampling),
        ("real_lhs", LatinHypercubeSampling),
        ("bin_random", BinaryRandomSampling),
        ("int_random", IntegerFromFloatSampling, {'clazz': FloatRandomSampling}),
        ("int_lhs", IntegerFromFloatSampling, {'clazz': LatinHypercubeSampling}),
        ("perm_random", PermutationRandomSampling)
    ]

    return SAMPLING


def get_sampling(name, *args, d={}, **kwargs):
    return get_from_list(get_sampling_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Selection
# =========================================================================================================

def get_selection_options():
    from pymoo.operators.selection.random_selection import RandomSelection
    from pymoo.operators.selection.tournament_selection import TournamentSelection

    SELECTION = [
        ("random", RandomSelection),
        ("tournament", TournamentSelection)
    ]

    return SELECTION


def get_selection(name, *args, d={}, **kwargs):
    return get_from_list(get_selection_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Crossover
# =========================================================================================================

def get_crossover_options():
    from pymoo.operators.crossover.differental_evolution_crossover import DifferentialEvolutionCrossover
    from pymoo.operators.crossover.exponential_crossover import ExponentialCrossover
    from pymoo.operators.crossover.half_uniform_crossover import HalfUniformCrossover
    from pymoo.operators.crossover.point_crossover import PointCrossover
    from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
    from pymoo.operators.crossover.uniform_crossover import UniformCrossover
    from pymoo.operators.integer_from_float_operator import IntegerFromFloatCrossover
    from pymoo.operators.crossover.edge_recombination_crossover import EdgeRecombinationCrossover
    from pymoo.operators.crossover.order_crossover import OrderCrossover

    CROSSOVER = [
        ("real_sbx", SimulatedBinaryCrossover, dict(prob=0.9, eta=30)),
        ("int_sbx", IntegerFromFloatCrossover, dict(clazz=SimulatedBinaryCrossover, prob=0.9, eta=30)),
        ("real_de", DifferentialEvolutionCrossover),
        ("(real|bin|int)_ux", UniformCrossover),
        ("(bin|int)_hux", HalfUniformCrossover),
        ("(real|bin|int)_exp", ExponentialCrossover),
        ("(real|bin|int)_one_point", PointCrossover, {'n_points': 1}),
        ("(real|bin|int)_two_point", PointCrossover, {'n_points': 2}),
        ("(real|bin|int)_k_point", PointCrossover),
        ("perm_ox", OrderCrossover),
        ("perm_erx", EdgeRecombinationCrossover)
    ]

    return CROSSOVER


def get_crossover(name, *args, d={}, **kwargs):
    return get_from_list(get_crossover_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Mutation
# =========================================================================================================

def get_mutation_options():
    from pymoo.operators.mutation.no_mutation import NoMutation
    from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
    from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
    from pymoo.operators.integer_from_float_operator import IntegerFromFloatMutation
    from pymoo.operators.mutation.inversion_mutation import InversionMutation

    MUTATION = [
        ("none", NoMutation, {}),
        ("real_pm", PolynomialMutation, dict(eta=20)),
        ("int_pm", IntegerFromFloatMutation, dict(clazz=PolynomialMutation, eta=20)),
        ("bin_bitflip", BinaryBitflipMutation),
        ("perm_inv", InversionMutation)
    ]

    return MUTATION


def get_mutation(name, *args, d={}, **kwargs):
    return get_from_list(get_mutation_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Termination
# =========================================================================================================

def get_termination_options():
    from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
    from pymoo.util.termination.max_gen import MaximumGenerationTermination
    from pymoo.util.termination.max_time import TimeBasedTermination
    from pymoo.util.termination.min_igd import IGDTermination
    from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination
    from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
    from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination
    from pymoo.util.termination.default import MultiObjectiveDefaultTermination, SingleObjectiveDefaultTermination

    TERMINATION = [
        ("n_eval", MaximumFunctionCallTermination),
        ("(n_gen|n_iter)", MaximumGenerationTermination),
        ("igd", IGDTermination),
        ("time", TimeBasedTermination),
        ("(x_tol|xtol)", DesignSpaceToleranceTermination),
        ("(f_tol$|ftol$)", MultiObjectiveSpaceToleranceTermination),
        ("(f_tol_s|ftol_s)", SingleObjectiveSpaceToleranceTermination),
        ("(default$|default_multi)", MultiObjectiveDefaultTermination),
        ("default_single$", SingleObjectiveDefaultTermination)
    ]

    return TERMINATION


def get_termination(name, *args, d={}, **kwargs):
    return get_from_list(get_termination_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Problems
# =========================================================================================================

def get_problem_options():
    PROBLEM = [
        ('ackley', Ackley),
        ('bnh', BNH),
        ('carside', Carside),
        ('ctp1', CTP1),
        ('ctp2', CTP2),
        ('ctp3', CTP3),
        ('ctp4', CTP4),
        ('ctp5', CTP5),
        ('ctp6', CTP6),
        ('ctp7', CTP7),
        ('ctp8', CTP8),
        ('dascmop1', DASCMOP1),
        ('dascmop2', DASCMOP2),
        ('dascmop3', DASCMOP3),
        ('dascmop4', DASCMOP4),
        ('dascmop5', DASCMOP5),
        ('dascmop6', DASCMOP6),
        ('dascmop7', DASCMOP7),
        ('dascmop8', DASCMOP8),
        ('dascmop9', DASCMOP9),
        ('mw1', MW1),
        ('mw2', MW2),
        ('mw3', MW3),
        ('mw4', MW4),
        ('mw5', MW5),
        ('mw6', MW6),
        ('mw7', MW7),
        ('mw8', MW8),
        ('mw9', MW9),
        ('mw10', MW10),
        ('mw11', MW11),
        ('mw12', MW12),
        ('mw13', MW13),
        ('mw14', MW14),
        ('dtlz1^-1', InvertedDTLZ1),
        ('dtlz1', DTLZ1),
        ('dtlz2', DTLZ2),
        ('dtlz3', DTLZ3),
        ('dtlz4', DTLZ4),
        ('dtlz5', DTLZ5),
        ('dtlz6', DTLZ6),
        ('dtlz7', DTLZ7),
        ('convex_dtlz2', ConvexDTLZ2),
        ('convex_dtlz4', ConvexDTLZ4),
        ('sdtlz1', ScaledDTLZ1),
        ('c1dtlz1', C1DTLZ1),
        ('c1dtlz3', C1DTLZ3),
        ('c2dtlz2', C2DTLZ2),
        ('c3dtlz1', C3DTLZ1),
        ('c3dtlz4', C3DTLZ4),
        ('dc1dtlz1', DC1DTLZ1),
        ('dc1dtlz3', DC1DTLZ3),
        ('dc2dtlz1', DC2DTLZ1),
        ('dc2dtlz3', DC2DTLZ3),
        ('dc3dtlz1', DC3DTLZ1),
        ('dc3dtlz3', DC3DTLZ3),
        ('cantilevered_beam', CantileveredBeam),
        ('griewank', Griewank),
        ('himmelblau', Himmelblau),
        ('knp', Knapsack),
        ('kursawe', Kursawe),
        ('osy', OSY),
        ('pressure_vessel', PressureVessel),
        ('rastrigin', Rastrigin),
        ('rosenbrock', Rosenbrock),
        ('schwefel', Schwefel),
        ('sphere', Sphere),
        ('tnk', TNK),
        ('truss2d', Truss2D),
        ('welded_beam', WeldedBeam),
        ('zakharov', Zakharov),
        ('zdt1', ZDT1),
        ('zdt2', ZDT2),
        ('zdt3', ZDT3),
        ('zdt4', ZDT4),
        ('zdt5', ZDT5),
        ('zdt6', ZDT6),
        ('g01', G1),
        ('g02', G2),
        ('g03', G3),
        ('g04', G4),
        ('g05', G5),
        ('g06', G6),
        ('g07', G7),
        ('g08', G8),
        ('g09', G9),
        ('g10', G10),
        ('wfg1', WFG1),
        ('wfg2', WFG2),
        ('wfg3', WFG3),
        ('wfg4', WFG4),
        ('wfg5', WFG5),
        ('wfg6', WFG6),
        ('wfg7', WFG7),
        ('wfg8', WFG8),
        ('wfg9', WFG9)
    ]

    return PROBLEM


def get_problem(name, *args, d={}, **kwargs):
    if name.startswith("go-"):
        from pymoo.vendor.global_opt import get_global_optimization_problem_options
        return get_from_list(get_global_optimization_problem_options(), name.lower(), args, {**d, **kwargs})
    else:
        return get_from_list(get_problem_options(), name.lower(), args, {**d, **kwargs})


# =========================================================================================================
# Weights
# =========================================================================================================

def get_reference_direction_options():
    from pymoo.util.reference_direction import UniformReferenceDirectionFactory
    from pymoo.util.reference_direction import MultiLayerReferenceDirectionFactory
    from pymoo.util.ref_dirs.reduction import ReductionBasedReferenceDirectionFactory
    from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
    from pymoo.util.ref_dirs.energy_layer import LayerwiseRieszEnergyReferenceDirectionFactory

    REFERENCE_DIRECTIONS = [
        ("(das-dennis|uniform)", UniformReferenceDirectionFactory),
        ("multi-layer", MultiLayerReferenceDirectionFactory),
        ("(energy|riesz)", RieszEnergyReferenceDirectionFactory),
        ("(layer-energy|layer-riesz)", LayerwiseRieszEnergyReferenceDirectionFactory),
        ("red", ReductionBasedReferenceDirectionFactory)
    ]

    return REFERENCE_DIRECTIONS


def get_reference_directions(name, *args, d={}, **kwargs):
    return get_from_list(get_reference_direction_options(), name, args, {**d, **kwargs}).do()


# =========================================================================================================
# Visualization
# =========================================================================================================

def get_visualization_options():
    from pymoo.visualization.pcp import PCP
    from pymoo.visualization.petal import Petal
    from pymoo.visualization.radar import Radar
    from pymoo.visualization.radviz import Radviz
    from pymoo.visualization.scatter import Scatter
    from pymoo.visualization.star_coordinate import StarCoordinate
    from pymoo.visualization.heatmap import Heatmap
    from pymoo.visualization.fitness_landscape import FitnessLandscape

    VISUALIZATION = [
        ("scatter", Scatter),
        ("heatmap", Heatmap),
        ("pcp", PCP),
        ("petal", Petal),
        ("radar", Radar),
        ("radviz", Radviz),
        ("star", StarCoordinate),
        ("fitness-landscape", FitnessLandscape)
    ]

    return VISUALIZATION


def get_visualization(name, *args, d={}, **kwargs):
    return get_from_list(get_visualization_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Performance Indicator
# =========================================================================================================


def get_performance_indicator_options():
    from pymoo.performance_indicator.gd import GD
    from pymoo.performance_indicator.gd_plus import GDPlus
    from pymoo.performance_indicator.igd import IGD
    from pymoo.performance_indicator.igd_plus import IGDPlus
    from pymoo.performance_indicator.hv import Hypervolume
    from pymoo.performance_indicator.rmetric import RMetric

    PERFORMANCE_INDICATOR = [
        ("gd", GD),
        ("gd+", GDPlus),
        ("igd", IGD),
        ("igd+", IGDPlus),
        ("hv", Hypervolume),
        ("rmetric", RMetric)
    ]
    return PERFORMANCE_INDICATOR


def get_performance_indicator(name, *args, d={}, **kwargs):
    return get_from_list(get_performance_indicator_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# DECOMPOSITION
# =========================================================================================================

def get_decomposition_options():
    from pymoo.decomposition.pbi import PBI
    from pymoo.decomposition.tchebicheff import Tchebicheff
    from pymoo.decomposition.weighted_sum import WeightedSum
    from pymoo.decomposition.asf import ASF
    from pymoo.decomposition.aasf import AASF
    from pymoo.decomposition.perp_dist import PerpendicularDistance

    DECOMPOSITION = [
        ("weighted-sum", WeightedSum),
        ("tchebi", Tchebicheff),
        ("pbi", PBI),
        ("asf", ASF),
        ("aasf", AASF),
        ("perp_dist", PerpendicularDistance)
    ]

    return DECOMPOSITION


def get_decomposition(name, *args, d={}, **kwargs):
    return get_from_list(get_decomposition_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# DECOMPOSITION
# =========================================================================================================

def get_decision_making_options():
    from pymoo.decision_making.high_tradeoff import HighTradeoffPoints
    from pymoo.decision_making.pseudo_weights import PseudoWeights

    DECISION_MAKING = [
        ("high-tradeoff", HighTradeoffPoints),
        ("pseudo-weights", PseudoWeights)
    ]

    return DECISION_MAKING


def get_decision_making(name, *args, d={}, **kwargs):
    return get_from_list(get_decision_making_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Documentation
# =========================================================================================================


def dummy(name, kwargs):
    """
    A convenience method to get a {type} object just by providing a string.

    Parameters
    ----------

    name : {{ {options} }}
        Name of the {type}.

    kwargs : dict
        Dictionary that should be used to call the method mapped to the {type} factory function.

    Returns
    -------
    class : {clazz}
        An {type} object based on the string. `None` if the {type} was not found.

    """
    pass


def options_to_string(l):
    return ", ".join(["'%s'" % k[0] for k in l])


if Configuration.parse_custom_docs:
    from pymoo.docs import parse_doc_string

    from pymoo.factory import get_algorithm_options, get_selection_options, get_crossover_options, \
        get_mutation_options, get_termination_options, get_algorithm, get_selection, get_crossover, get_mutation, \
        get_termination, get_sampling, get_sampling_options

    parse_doc_string(dummy, get_algorithm, {"type": "algorithm",
                                            "clazz": ":class:`~pymoo.model.algorithm.Algorithm`",
                                            "options": options_to_string(get_algorithm_options())
                                            })

    parse_doc_string(dummy, get_sampling, {"type": "sampling",
                                           "clazz": ":class:`~pymoo.model.sampling.Sampling`",
                                           "options": options_to_string(get_sampling_options())
                                           })

    parse_doc_string(dummy, get_selection, {"type": "selection",
                                            "clazz": ":class:`~pymoo.model.selection.Selection`",
                                            "options": options_to_string(get_selection_options())
                                            })

    parse_doc_string(dummy, get_crossover, {"type": "crossover",
                                            "clazz": ":class:`~pymoo.model.crossover.Crossover`",
                                            "options": options_to_string(get_crossover_options())
                                            })

    parse_doc_string(dummy, get_mutation, {"type": "mutation",
                                           "clazz": ":class:`~pymoo.model.mutation.Mutation`",
                                           "options": options_to_string(get_mutation_options())
                                           })

    parse_doc_string(dummy, get_termination, {"type": "termination",
                                              "clazz": ":class:`~pymoo.model.termination.termination`",
                                              "options": options_to_string(get_termination_options())
                                              })
