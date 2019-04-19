# -*- encoding: utf-8 -*-

import inspect
import re

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
                :class:`~pymoo.model.sampling.Sampling`, :class:`~pymoo.model.population.Population`, :obj:`numpy.array`
                    The sampling process defines the initial set of solutions which are the starting point of the
                    optimization algorithm. Here, you have three different options by passing

                        (i) A :class:`~pymoo.model.sampling.Sampling` implementation which is an implementation of a 
                        random sampling method. 

                        (ii) A :class:`~pymoo.model.population.Population` object containing the variables to
                        be evaluated initially OR already evaluated solutions (F needs to be set in this case).

                        (iii) Pass a two dimensional :obj:`numpy.array` with (n_individuals, n_var) which contains the variable 
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
                    The crossover has the purpose of create offsprings during the evolution. After the mating selection
                    the parents are passed to the crossover operator which will dependent on the implementation create
                    a different number of offsprings.
                """,

    "mutation": """:class:`~pymoo.model.mutation.Mutation`
                    Some genetic algorithms rely only on the mutation operation. However, it has shown that increases
                    the performance to perform a mutation after creating the offsprings through crossover as well.
                    Usually the mutation operator needs to be initialized with a probability to be executed. 
                    Having a high probability of mutation will most of the time increase the diversity in the population.
                """,

    "survival": """:class:`~pymoo.model.survival.Survival`
                    The survival selection is the key for many genetic algorithms. It is responsible to define the
                    goal of convergence by choosing the individual to survive or be truncated each generation.
                    For single-objective problems a selection based on the fitness is used commonly. However, for 
                    multi-objective problems different concepts are introduced.
                """,

    "ref_points": """:obj:`numpy.array`
                    Reference Points (or also called Aspiration Points) as a :obj:`numpy.array` where each row 
                    represents a point and each column a variable (must be equal to the objective dimension of the problem)
                """,

    "eliminate_duplicates": """bool
                    The genetic algorithm implementation has a built in feature that eliminates duplicates after merging
                    the parent and the offspring population. If there are duplicates with respect to the current 
                    population or in the offsprings itself they are removed and the mating process is repeated to
                    fill up the offsprings until the desired number of unique offsprings is met.            
                """,

    "n_offsprings": """int (default: None)
                    Number of offspring that are created through mating. By default *n_offsprings=None* which
                    sets the number of offsprings equal to the population size. By setting *n_offsprings=1* a, so called,
                    steady-state version of an algorithm can be achieved.
            """,

    "ref_dirs": """:obj:`numpy.array`
                    The reference direction that should be used during the optimization. Each row represents a reference line
                    and each column a variable.
            """

}


# =========================================================================================================
# Util for docstrings
# =========================================================================================================

def parse_doc_string(source, dest=None, other={}):

    if dest is None:
        dest = source

    D = {k: v.strip() for k, v in docs.items()}
    _doc = source.__doc__.format(**{**D, **other})

    lines = inspect.getsource(source)

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

    dest.__doc__ = signature + "\n" + _doc


