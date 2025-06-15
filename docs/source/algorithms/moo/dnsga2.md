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

.. _nb_dnsga2:
```

# D-NSGA-II: Dynamic Multi-Objective Optimization Using Modified NSGA-II

+++

The algorithm is implemented based on <cite data-cite="dnsga2"></cite>. D-NSGA-II modifies the commonly-used NSGA-II procedure in tracking a new Pareto-optimal front as soon as there is a change in the problem. The introduction of a few random solutions or a few mutated solutions provides some diversity and gives the algorithm a chance to escape from a local optimum over time.

```{code-cell} ipython3
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.core.callback import CallbackCollection
from pymoo.optimize import minimize
from pymoo.problems.dyn import TimeSimulation
from pymoo.problems.dynamic.df import DF1

from pymoo.visualization.video.callback_video import ObjectiveSpaceAnimation

problem = DF1(taut=2, n_var=2)

algorithm = DNSGA2(version="A")

res = minimize(problem,
               algorithm,
               termination=('n_gen', 100),
               callback=TimeSimulation(),
               seed=1,
               verbose=False)
```
