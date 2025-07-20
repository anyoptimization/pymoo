---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw-cell}
.. _nb_survival:
```

## Survival

+++

### Rank and Crowding

+++

The original survival strategy proposed in [NSGA-II](../algorithms/moo/nsga2.ipynb) <cite data-cite="nsga2"></cite> ranks solutions in fronts by dominance criterion and uses a diversity metric denoted crowding distances to sort individuals in each front. This is used as criterion to compare individuals in elitist parent selection schemes and to truncate the population in the survival selection stage of algorithms.

Variants of the original algorithm have been proposed in the literature to address different performance aspects. Therefore the class ``RankAndCrowding`` from pymoo is a generalization of NSGA-II's survival in which several crowding metrics can be used. Some are already implemented and can be parsed as strings in the ``crowding_func`` argument, while others might be defined from scratch and parsed as callables. The ones available are:

- **Crowding Distance** (*'cd'*): Proposed by Deb et al. <cite data-cite="nsga2"></cite> in NSGA-II.
- **Pruning Crowding Distance** (*'pruning-cd'* or *'pcd'*): Proposed by Kukkonen & Deb <cite data-cite="gde3pruning"></cite>, it recursively recalculates crowding distances as removes individuals from a population to improve diversity.
- ***M*-Nearest Neighbors** (*'mnn'*): Proposed by Kukkonen & Deb <cite data-cite="gde3many"></cite> in an extension of GDE3 to many-objective problems.
- **2-Nearest Neighbors** (*'2nn'*): Also proposed by Kukkonen & Deb <cite data-cite="gde3many"></cite>, it is a variant of M-Nearest Neighbors in which the number of neighbors is two.
- **Crowding Entropy** (*'ce'*): Proposed by Wang et al. <cite data-cite="mosade"></cite> it considers the relative position of a solution between its neighors.

We encourage users to try ``crowding_func='pcd'`` for two-objective problems and ``crowding_func='mnn'`` for problems with more than two objectives.

If callable, it has the form ``fun(F, filter_out_duplicates=None, n_remove=None, **kwargs)`` in which F (n, m) and must return metrics in a (n,) array.

The ``ConstrRankAndCrowding`` class has the constraint handling approach proposed by Kukkonen, S. & Lampinen, J. <cite data-cite="gde3"></cite> implemented in which solutions are also sorted in constraint violations space.

+++

In the following examples the code for plotting was omitted.

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from plots import plot_pairs_2d, plot_pairs_3d
```

```{code-cell} ipython3
# Problem definition Truss-2d - a two-objective problem
problem = get_problem("truss2d")

# Algorithms
nsga2 = NSGA2(70, survival=RankAndCrowding(crowding_func="cd"))
nsga2_p = NSGA2(70, survival=RankAndCrowding(crowding_func="pcd"))

# Minimization results
res_nsga2 = minimize(
    problem,
    nsga2,
    ('n_gen', 200),
    seed=12,
)

# Minimization results
res_nsga2_p = minimize(
    problem,
    nsga2_p,
    ('n_gen', 200),
    seed=12,
)
```

```{code-cell} ipython3
plot_pairs_2d(
    ("NSGA-II (original)", res_nsga2.F),
    ("NSGA-II (pruning)", res_nsga2_p.F),
    figsize=[12, 5],
    dpi=100,
)
```
