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

.. _nb_nsde:
```

.. meta::
   :description: An implementation of NSDE, a multi-objective optimization algorithm that combines NSGA-II's non-dominated sorting survival with differential evolution variation operators.

+++

.. meta::
   :keywords: NSDE, Differential Evolution, NSGA-II, Multi-objective Optimization, Python

+++

# NSDE: Non-dominated Sorting Differential Evolution

+++

NSDE keeps NSGA-II's non-dominated sorting and crowding-distance survival but replaces the simulated binary crossover and polynomial mutation with **differential evolution** variation. The DE `variant` (for example `DE/rand/1/bin`), the crossover rate `CR`, and the scaling factor `F` control how trial vectors are produced. This is often effective on continuous multi-objective problems where DE's difference-vector mutation converges quickly.

+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.moo.nsde import NSDE
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = NSDE(pop_size=100, variant="DE/rand/1/bin", CR=0.7)

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

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.nsde.NSDE
    :noindex:
```

+++

### References

Based on NSGA-II (Deb et al., 2002) with differential evolution variation (Storn & Price, 1997). See also Kukkonen & Lampinen (2005) for generalized differential evolution.
