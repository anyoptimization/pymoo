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

.. _nb_gde3:
```

.. meta::
   :description: An implementation of Generalized Differential Evolution 3 (GDE3), a multi-objective optimization algorithm that combines differential evolution variation with non-dominated sorting based survival.

+++

.. meta::
   :keywords: GDE3, Differential Evolution, Multi-objective Optimization, Non-dominated Sorting, Python

+++

# GDE3: Generalized Differential Evolution 3

+++

GDE3 extends Differential Evolution (DE) to multi-objective optimization. It generates offspring with the usual DE mutation and crossover (the `variant` controls the strategy, e.g. `DE/rand/1/bin`) and then selects survivors with NSGA-II's non-dominated sorting and crowding distance. A one-to-one greedy comparison between a parent and its trial vector handles the cases where one dominates the other, which improves convergence on continuous problems.

+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.moo.gde3 import GDE3
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = GDE3(pop_size=100, variant="DE/rand/1/bin", CR=0.5)

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

pymoo also ships the GDE3 variants ``GDE3MNN``, ``GDE32NN``, and ``GDE3PCD``, which replace the crowding metric used during survival with an alternative diversity estimator.

+++

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.gde3.GDE3
    :noindex:
```

+++

### References

Kukkonen, S., & Lampinen, J. (2005). *GDE3: The third evolution step of generalized differential evolution.* IEEE Congress on Evolutionary Computation (CEC), 443-450.
