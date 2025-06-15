---
jupytext:
  formats: ipynb,md:myst
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
---
editable: true
raw_mimetype: text/restructuredtext
slideshow:
  slide_type: ''
---
.. _nb_matrix_inversion:
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Matrix Inversion

In this case study, the optimization of a matrix will be illustrated. Of course, we all know that there are very efficient algorithms for calculating an inverse of a matrix. However, for the sake of illustration, a small example will show that pymoo can also be used to optimize matrices or even tensors.

Assuming matrix `A` has a size of `n x n`, the problem can be defined by optimizing a vector consisting of `n**2` variables. During evaluation the vector `x` is reshaped to inversion of the matrix to be found (and also stored as the attribute `A_inv` to be retrieved later).

```{code-cell} ipython3
from pymoo.core.problem import ElementwiseProblem


class MatrixInversionProblem(ElementwiseProblem):

    def __init__(self, A, **kwargs):
        self.A = A
        self.n = len(A)
        super().__init__(n_var=self.n**2, n_obj=1, xl=-100.0, xu=+100.0, **kwargs)


    def _evaluate(self, x, out, *args, **kwargs):
        A_inv = x.reshape((self.n, self.n))
        out["A_inv"] = A_inv

        I = np.eye(self.n)
        out["F"] = ((I - (A @ A_inv)) ** 2).sum()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Now let us see what solution is found to be optimal

```{code-cell} ipython3
import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize

np.random.seed(1)
A = np.random.random((2, 2))

problem = MatrixInversionProblem(A)

algorithm = DE()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

opt = res.opt[0]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In this case the true optimum is actually known. It is

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
np.linalg.inv(A)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Let us see if the black-box optimization algorithm has found something similar

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
opt.get("A_inv")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This small example illustrates how a matrix can be optimized. In fact, this is implemented by optimizing a vector of variables that are reshaped during evaluation.
