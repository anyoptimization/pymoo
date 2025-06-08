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
pycharm:
  name: '#%% raw

    '
raw_mimetype: text/restructuredtext
---
.. _nb_parallelization_vectorized:
```

+++ {"pycharm": {"name": "#%% md\n"}}

# Vectorized Matrix Operations

+++ {"pycharm": {"name": "#%% md\n"}}

One way is using the `NumPy` matrix operations, which has been used for almost all test problems implemented in *pymoo*.
By default, `elementwise_evaluation` is set to `False`, which implies the `_evaluate` retrieves a set of solutions.
Thus, `x` is a matrix where each row is an individual, and each column a variable.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
import numpy as np
from pymoo.core.problem import Problem

class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
         out["F"] = np.sum(x ** 2, axis=1)

problem = MyProblem()
```

+++ {"pycharm": {"name": "#%% md\n"}}

The `axis=1` operation parallelizes the sum of the matrix directly using an efficient NumPy operation.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
print('Threads:', res.exec_time)
```
