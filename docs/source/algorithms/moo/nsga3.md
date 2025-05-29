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

.. _nb_nsga3:
```

.. meta::
   :description: NSGA-III (also known as NSGA3) has been proposed for many-objective optimization to address the shortcomings of its predecessor NSGA-II. Using reference directions, the algorithm scales to many objectives and achieves a diverse set of non-dominated solutions.

+++

.. meta::
   :keywords: NSGA3, NSGA-III, Many-objective Optimization, Reference Directions, Non-Dominated Sorting, Multi-objective Optimization, Python

+++

# NSGA-III


The algorithm is implemented based on <cite data-cite="nsga3-part1"></cite> <cite data-cite="nsga3-part2"></cite>. Implementation details of this algorithm can be found in <cite data-cite="nsga3-norm"></cite>. NSGA-III is based on [Reference Directions](../../misc/reference_directions.ipynb) which need to be provided when the algorithm is initialized. 

First, the non-dominated sorting is done as in NSGA-II for survival. 

+++

<div style="text-align: center;">
    <img src="https://github.com/anyoptimization/pymoo-data/blob/main/docs/images/nsga3_survival_1.png?raw=true" width="350">
</div>

+++

Second, from the splitting front, some solutions need to be selected. NSGA-III fills up the underrepresented reference direction first. If the reference direction does not have any solution assigned, then the solution with the smallest perpendicular distance in the normalized objective space is surviving. In case a second solution for this reference line is added, it is assigned randomly. 

+++

<div style="text-align: center;">
    <img src="https://github.com/anyoptimization/pymoo-data/blob/main/docs/images/nsga3_survival_2.png?raw=true" width="350">
</div>

+++

Thus, when this algorithm converges, each reference line seeks to find a good representative non-dominated solution.

+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga3 import NSGA3


from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# create the algorithm object
algorithm = NSGA3(pop_size=92,
                  ref_dirs=ref_dirs)

# execute the optimization
res = minimize(get_problem("dtlz1"),
               algorithm,
               seed=1,
               termination=('n_gen', 600))

Scatter().add(res.F).show()
```

```{code-cell} ipython3
res = minimize(get_problem("dtlz1^-1"),
               algorithm,
               seed=1,
               termination=('n_gen', 600))

Scatter().add(res.F).show()
```

+++ {"raw_mimetype": "text/restructuredtext"}

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.nsga3.NSGA3
    :noindex:
```
