---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_selection:
```

## Selection

This module defines the mating selection during the execution of a genetic algorithm. At the beginning of the mating process, parents need to be selected to be mated using the crossover operation.


+++

 
 
 
 

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_selection_random:
```

### Random Selection

+++

Here, we randomly pick solutions from the current population to be used for recombination. The implementation uses a permutation to avoid repetitive individuals. For instance, let us consider the case where only two parents are desired to be selected: The permutation (5,2,3,4,1,0), will lead to the parent selection of (5,2), (3,4), (1,0), where no parent can participate twice for mating.

```{code-cell} ipython3
from pymoo.operators.selection.rnd import RandomSelection

selection = RandomSelection()
```

 

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_selection_tournament:
```

### Tournament Selection

+++

It has been shown that tournament pressure is helpful for faster convergence. This implementation provides the functionality to define a tournament selection very generic. 
Below we show a binary tournament selection (two individuals are participating in each competition).

Having defined the number of participants, the winner needs to be written to an output array. Here, we use the fitness values (if constraints should be considered, CV should be added as well) to achieve that.

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.problems import get_problem


def binary_tournament(pop, P, _, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")

    # the result this function returns
    import numpy as np
    S = np.full(n_tournaments, -1, dtype=np.int)

    # now do all the tournaments
    for i in range(n_tournaments):
        a, b = P[i]

        # if the first individual is better, choose it
        if pop[a].F < pop[b].F:
            S[i] = a

        # otherwise take the other individual
        else:
            S[i] = b

    return S


selection = TournamentSelection(pressure=2, func_comp=binary_tournament)

problem = get_problem("rastrigin")

algorithm = GA(pop_size=100, eliminate_duplicates=True)

res = minimize(problem, algorithm, termination=('n_gen', 100), verbose=False)

print(res.X)
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autofunction:: pymoo.core.selection.Selection
    :noindex:
```
