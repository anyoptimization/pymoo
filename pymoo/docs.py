# -*- encoding: utf-8 -*-

import inspect
import re
from pymoo.config import Config

# =========================================================================================================
# Docstrings Dictionary
# =========================================================================================================


algorithms = {
    "pop_size": """ int
                The population sized used by the algorithm.
    """,

    "n_offsprings": """ int
                The number of offsprings that should be created in each generation.
    """,

    "sampling": """
                :class:`~pymoo.core.sampling.Sampling`, :class:`~pymoo.core.population.Population`, :obj:`numpy.array`
                    The sampling process defines the initial set of solutions which are the starting point of the
                    optimization algorithm. Here, you have three different options by passing

                        (i) A :class:`~pymoo.core.sampling.Sampling` implementation which is an implementation of a 
                        random sampling method. 

                        (ii) A :class:`~pymoo.core.population.Population` object containing the variables to
                        be evaluated initially OR already evaluated solutions (F needs to be set in this case).

                        (iii) Pass a two dimensional :obj:`numpy.array` with (n_individuals, n_var) which contains the variable 
                        space values for each individual.        
                """,

    "selection": """:class:`~pymoo.core.selection.Selection`
                    This object defines the mating selection to be used. 
                    In an evolutionary algorithm each generation parents need to be selected to produce new offsprings using 
                    different recombination and mutation operators. Different strategies for selecting parents are possible e.g. 
                    selecting them just randomly, only in the neighborhood, using a tournament selection to introduce some selection 
                    pressure, ... 
                    """,
    "crossover": """:class:`~pymoo.core.crossover.Crossover`
                    The crossover has the purpose of create offsprings during the evolution. After the mating selection
                    the parents are passed to the crossover operator which will dependent on the implementation create
                    a different number of offsprings.
                """,

    "mutation": """:class:`~pymoo.core.mutation.Mutation`
                    Some genetic algorithms rely only on the mutation operation. However, it has shown that increases
                    the performance to perform a mutation after creating the offsprings through crossover as well.
                    Usually the mutation operator needs to be initialized with a probability to be executed. 
                    Having a high probability of mutation will most of the time increase the diversity in the population.
                """,

    "survival": """:class:`~pymoo.core.survival.Survival`
                    The survival selection is the key for many genetic algorithms. It is responsible to define the
                    goal of convergence by choosing the individual to survive or be truncated each generation.
                    For single-objective single a selection based on the fitness is used commonly. However, for 
                    multi-objective single different concepts are introduced.
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

visualization = {
    "figsize": """tuple
                The figure size. Default (figsize=(8, 6)). For some plots changing the size might have side effects for position.
                """,

    "title": """str or tuple
                The title of the figure. If some additional kwargs should be provided this can be achieved by providing a tuple
                ("name", {"key" : val}).
                """,

    "legend": """str
                Whether a legend should be shown or not.
                """,

    "tight_layout": """bool
                        Whether tight layout should be used.
                    """,

    "bounds": """tuple
                If plot requires normalization, it might be necessary to supply the boundaries. (Otherwise they might be
                approximate by the minimum and maximum of the provided data). The boundaries should be provided as a list/tuple or
                2D numpy array, where the first element represents the minimum, second the second the maximum values.
                If only an integer or float is supplied, the boundaries apply for each variable.
            """,

    "reverse": """bool
                    If plot requires normalization, then the reverse values can be plotted (1 - Input). For some plots
                    it can be useful to interpret a larger area as better regarding a value. If minimization applies, a smaller
                    area means better, which can be misleading.
                """,

    "axis_style": """dict
                        Most of the plots consists of an axis. The style of the axis, e.g. color, alpha, ..., can be changed to
                        further modify the plot appealing.
                    """,

    "cmap": """colormap
                    For some plots different kind of colors are used. The colormap can be changed to modify the color sequence
                    for the plots.
            """,

    "labels": """str or list
                    The labels to be used for each variable provided in the plot. If a string is used, then they will
                    be enumerated. Otherwise, a list equal to the number of variables can be provided directly.
            """,

    "func_number_to_text": """func
                                A function which defines how numerical values should be represented if present in the plot 
                                for instance scientific notation, rounding and so on.
                            """,

}

docs = {**algorithms, **visualization}


# =========================================================================================================
# Util for docstrings
# =========================================================================================================

def parse_doc_string(source, dest=None, other={}):
    if not Config.parse_custom_docs:
        return

    if dest is None:
        dest = source

    D = {k: v.strip() for k, v in docs.items()}
    
    doc = source.__doc__
    

    lines = inspect.getsource(source)
    if doc is not None:
        
        doc = doc.format(**{**D, **other})

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

        if dest is not None:
            dest.__doc__ = signature + "\n" + doc
