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

.. _nb_nelder_mead:
```

# Nelder Mead


This algorithm is implemented based on <cite data-cite="NelderMead65"></cite>. In addition to other implementations, a boundary check is included. This ensures that the search considers the box constraints of the given optimization problem. If no boundaries are provided, the algorithm falls back to a search without any constraints. 

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("sphere")

algorithm = NelderMead()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.soo.nonconvex.nelder.NelderMead
    :noindex:
```
