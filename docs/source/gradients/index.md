---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: default
  language: python
  name: default
---

```{raw-cell}
---
editable: true
raw_mimetype: text/restructuredtext
slideshow:
  slide_type: ''
---
.. _nb_gradients:
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Gradients


If the problem is implemented using autograd, then the gradients through automatic differentiation are available out of the box. Let us consider the following problem definition for a simple quadratic function without any constraints:

```{code-cell} ipython3
import numpy as np
import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem
from pymoo.gradient.automatic import AutomaticDifferentiation


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=10, n_obj=1, xl=-5, xu=5)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = anp.sum(anp.power(x, 2), axis=1)

problem = AutomaticDifferentiation(MyProblem())
```

The gradients can be retrieved by appending `F` to the `return_values_of` parameter:

```{code-cell} ipython3
X = np.array([np.arange(10)]).astype(float)
F, dF = problem.evaluate(X, return_values_of=["F", "dF"])
```

The resulting gradients are stored in `dF` and the shape is (n_rows, n_objective, n_vars):

```{code-cell} ipython3
print(X, F)
print(dF.shape)
print(dF)
```

Analogously, the gradient of constraints can be retrieved by appending `dG`.
