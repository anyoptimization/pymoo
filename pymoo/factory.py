# -*- encoding: utf-8 -*-

from pymoo.algorithms.nsga3 import NSGA3, ReferenceDirectionSurvival, comp_by_cv_then_random

from pymoo.algorithms.nsga2 import NSGA2, RankAndCrowdingSurvival, binary_tournament

from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.sampling.random_sampling import RandomSampling
import inspect, re


# =========================================================================================================
# Util for docstrings
# =========================================================================================================

def parse_doc_string(obj):
    D = {k: v.strip() for k, v in docs.items()}
    _doc = obj.__doc__.format(**D)

    lines = inspect.getsource(obj)

    cnt = 0
    b = False

    for i, c in enumerate(lines):
        if b and cnt == 0:
            break
        if c == "(":
            cnt += 1
            b = True
        elif c == ")":
            cnt -= 1

    signature = lines[:i]
    signature = re.sub(r"[\n\t]*", "", signature)
    signature = re.sub(r"\s+", " ", signature)
    signature = re.sub(r"def\s*", "", signature)
    signature = signature.strip()

    obj.__doc__ = signature + "\n" + _doc


# =========================================================================================================
# Docstrings Dictionary
# =========================================================================================================


docs = {
    "pop_size": """ int
                The population sized used by the algorithm.
    """,

    "n_offsprings": """ int
                The number of offsprings that should be created in each generation.
    """,

    "sampling": """
                :class:`~pymoo.model.sampling.Sampling`, :class:`~pymoo.model.population.Population`, np.array 
                    The sampling process defines the initial set of solutions which are the starting point of the
                    optimization algorithm. Here, you have three different options by passing
                    
                        (i) A :class:`~pymoo.model.sampling.Sampling` implementation which is an implementation of a 
                        random sampling method. 
                        
                        (ii) A :class:`~pymoo.model.population.Population` object containing the variables to
                        be evaluated initially OR already evaluated solutions (F needs to be set in this case).
                        
                        (iii) Pass a two dimensional np.array with (n_individuals, n_var) which contains the variable 
                        space values for each individual.        
                """,

    "selection": """:class:`~pymoo.model.selection.Selection`
                    This object defines the mating selection to be used. 
                    In an evolutionary algorithm each generation parents need to be selected to produce new offsprings using 
                    different recombination and mutation operators. Different strategies for selecting parents are possible e.g. 
                    selecting them just randomly, only in the neighbourhood, using a tournament selection to introduce some seletion 
                    pressure, ... 
                    """,
    "crossover": """:class:`~pymoo.model.crossover.Crossover`
                
                """,

    "mutation": """:class:`~pymoo.model.mutation.Mutation`

                """,

    "survival": """:class:`~pymoo.model.survival.Survival`

                """,

    "eliminate_duplicates": """bool
                    The genetic algorithm implementation has a built in feature that eliminates duplicates after merging
                    the parent and the offspring population. If there are duplicates with respect to the current 
                    population or in the offsprings itself they are removed and the mating process is repeated to
                    fill up the offsprings until the desired number of unique offsprings is met.            
                """,

    "ref_dirs": """np.array
                    The reference direction that should be used during the optimization. Each row represents a reference line
                    and each column a variable.
            """,

}


# =========================================================================================================
# Algorithms
# =========================================================================================================


def nsga2(
        pop_size=100,
        sampling=RandomSampling(),
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=SimulatedBinaryCrossover(prob_cross=0.9, eta_cross=15),
        mutation=PolynomialMutation(prob_mut=None, eta_mut=20),
        eliminate_duplicates=True,
        **kwargs):
    """

    Parameters
    ----------
    pop_size : {pop_size}
    sampling : {sampling}
    selection : {selection}
    crossover : {crossover}
    mutation : {mutation}
    eliminate_duplicates : {eliminate_duplicates}

    Returns
    -------
    nsga2 : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an NSGA2 algorithm object.


    """

    return NSGA2(pop_size=pop_size,
                 sampling=sampling,
                 selection=selection,
                 crossover=crossover,
                 mutation=mutation,
                 survival=RankAndCrowdingSurvival(),
                 eliminate_duplicates=eliminate_duplicates,
                 **kwargs)


def nsga3(
        ref_dirs,
        pop_size=None,
        sampling=RandomSampling(),
        selection=TournamentSelection(func_comp=comp_by_cv_then_random),
        crossover=SimulatedBinaryCrossover(prob_cross=1.0, eta_cross=30),
        mutation=PolynomialMutation(prob_mut=None, eta_mut=20),
        eliminate_duplicates=True,
        **kwargs):
    """

    Parameters
    ----------
    ref_dirs : {ref_dirs}
    pop_size : int (default = None)
        By default the population size is set to None which means that it will be equal to the number of reference
        line. However, if desired this can be overwritten by providing a positve number.
    sampling : {sampling}
    selection : {selection}
    crossover : {crossover}
    mutation : {mutation}
    eliminate_duplicates : {eliminate_duplicates}

    Returns
    -------
    nsga3 : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an NSGA3 algorithm object.


    """

    return NSGA3(ref_dirs,
                 pop_size=pop_size,
                 sampling=sampling,
                 selection=selection,
                 crossover=crossover,
                 mutation=mutation,
                 survival=ReferenceDirectionSurvival(ref_dirs),
                 eliminate_duplicates=eliminate_duplicates,
                 **kwargs)


# =========================================================================================================
# Selection
# =========================================================================================================


# =========================================================================================================
# Crossover
# =========================================================================================================


# =========================================================================================================
# Mutation
# =========================================================================================================


parse_doc_string(nsga2)
parse_doc_string(nsga3)
