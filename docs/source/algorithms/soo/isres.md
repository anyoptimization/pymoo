---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: ''
  display_name: ''
---

```{raw-cell}
---
raw_mimetype: text/restructuredtext
editable: true
slideshow:
  slide_type: ''
---
.. _nb_isres:
```

.. meta::
   :description: The stochastic ranking is based on bubble sort and provides infeasible solutions a chance to survive during the environment selection. Adding this selection to an evolution strategy method has shown to be an effective optimization method: Stochastic Ranking Evolutionary Strategy.

+++

.. meta::
   :keywords: Improved Stochastic Ranking Evolutionary Strategy, ISRES,  Constrained Optimization, Real-Valued Optimization, Single-objective Optimization, Python

+++

# ISRES: Improved Stochastic Ranking Evolutionary Strategy

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Improved Stochastic Ranking Evolutionary Strategy <cite data-cite="isres"></cite>.

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("g1")

algorithm = ISRES(n_offsprings=200, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)

res = minimize(problem,
               algorithm,
               ("n_gen", 200),
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.soo.nonconvex.isres.ISRES
    :noindex:
```
