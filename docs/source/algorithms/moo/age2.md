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

.. _nb_agemoea2:
```

.. meta::
   :keywords: AGEMOEA2, NSGA-II, Non-Dominated Sorting, Multi-objective Optimization, Python

+++

# AGE-MOEA2: Adaptive Geometry Estimation based MOEA

+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = AGEMOEA2(pop_size=100)

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

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.age2.AGEMOEA2
    :noindex:
```
