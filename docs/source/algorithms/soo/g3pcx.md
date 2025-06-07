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

.. _nb_g3pcx:
```

# G3PCX: A Computationally Efficient Evolutionary Algorithm for Real-Parameter Optimization

+++

The algorithm is implemented based on <cite data-cite="g3pcx"></cite>.
This is an implementation of PCX operator using G3 model. This is an unconstrained optimization algorithm which is suitable for real parameter optimization. 

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.problems.single import Ackley
from pymoo.optimize import minimize

problem = Ackley()

algorithm = G3PCX()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.soo.nonconvex.g3pcx.G3PCX
    :noindex:
    :no-undoc-members:
```
