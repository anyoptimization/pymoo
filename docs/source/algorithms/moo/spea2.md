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

.. _nb_spea2:
```

.. meta::
   :description: An implementation of the Strength Pareto Evolutionary Algorithm 2 (SPEA2), a well-known multi-objective optimization algorithm that combines a strength-based fitness assignment with a density estimate and an environmental selection with archive truncation.

+++

.. meta::
   :keywords: SPEA2, Strength Pareto Evolutionary Algorithm, Multi-objective Optimization, Density Estimation, Python

+++

# SPEA2: Strength Pareto Evolutionary Algorithm 2

+++

SPEA2 is a well-established multi-objective evolutionary algorithm. It improves upon the original SPEA by combining a fine-grained, strength-based fitness assignment with a density estimate, and by using an environmental selection that keeps the number of non-dominated solutions stable through an archive truncation operator.

+++

## Key Features

**Strength-based fitness:** Each individual's raw fitness is the sum of the *strengths* of the solutions that dominate it, where an individual's strength is the number of solutions it dominates. Lower values are better, and all non-dominated solutions receive a raw fitness of zero.

**Density estimation:** To break ties among non-dominated solutions, a density term based on the distance to the *k*-th nearest neighbor in objective space is added to the raw fitness, promoting a well-spread front.

**Environmental selection with truncation:** A fixed-size archive of the best individuals is maintained across generations. When too many non-dominated solutions exist, a truncation operator iteratively removes the individual with the smallest distance to its neighbors, preserving diversity.

+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = SPEA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
```

+++

SPEA2 is a drop-in multi-objective algorithm and uses the same modular operators (sampling, crossover, mutation) as the other genetic algorithms in pymoo, so it can be customized in the same way.

+++

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.spea2.SPEA2
    :noindex:
```

+++

### References

Zitzler, E., Laumanns, M., & Thiele, L. (2001). *SPEA2: Improving the Strength Pareto Evolutionary Algorithm.* Technical Report 103, Computer Engineering and Networks Laboratory (TIK), ETH Zurich.
