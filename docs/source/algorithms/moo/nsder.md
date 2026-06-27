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
:raw_mimetype: text/restructuredtext

.. _nb_nsder:
```

.. meta::
   :description: An implementation of NSDE-R, a reference-direction based differential evolution algorithm for many-objective optimization.

+++

.. meta::
   :keywords: NSDE-R, NSDER, Differential Evolution, Reference Directions, Many-objective Optimization, Python

+++

# NSDE-R: Reference-direction based Differential Evolution

+++

NSDE-R brings **differential evolution** variation to many-objective optimization by combining it with the reference-direction based survival of NSGA-III. As with NSGA-III, a set of reference directions guides the selection so that the population stays well distributed across a high-dimensional objective space, while the DE operators (the `variant`, `CR`, and `F`) drive variation. The number of reference directions must match the number of objectives of the problem.

+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.moo.nsder import NSDER
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

problem = get_problem("dtlz2")

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithm = NSDER(ref_dirs=ref_dirs, variant="DE/rand/1/bin", CR=0.5)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

Scatter().add(res.F, facecolor="none", edgecolor="red").show()
```

+++

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.nsder.NSDER
    :noindex:
```

+++

### References

Reddy, S. R., & Dulikravich, G. S. (2019). *Many-objective differential evolution optimization based on reference points: NSDE-R.* Structural and Multidisciplinary Optimization, 60, 1455-1473.
