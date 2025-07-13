---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: default
  display_name: default
  language: python
---

```{raw-cell}
---
pycharm:
  name: '#%% raw

    '
raw_mimetype: text/restructuredtext
---
.. _nb_parallelization_starmap:
```

+++ {"pycharm": {"name": "#%% md\n"}}

# Starmap Interface

+++ {"pycharm": {"name": "#%% md\n"}}

In general, **pymoo** allows passing a `starmap` object to be used for parallelization. 
The `starmap` interface is defined in the Python standard library `multiprocessing.Pool.starmap` [function](https://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#multiprocessing.pool.Pool.starmap).
This allows for excellent and flexible parallelization opportunities. 

**IMPORTANT:** Please note that the problem needs to have set `elementwise_evaluation=True`, which indicates one call of `_evaluate` only takes care of a single solution.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
from pymoo.core.problem import ElementwiseProblem

class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
         out["F"] = (x ** 2).sum()
```

+++ {"pycharm": {"name": "#%% md\n"}}

Then, we can pass a `starmap` object to be used for parallelization.

+++ {"pycharm": {"name": "#%% md\n"}}

## Threads

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
from multiprocessing.pool import ThreadPool
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


# initialize the thread pool and create the runner
n_threads = 4
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

# define the problem by passing the starmap interface of the thread pool
problem = MyProblem(elementwise_runner=runner)

res = minimize(problem, GA(), termination=("n_gen", 50), seed=1)
print('Threads:', res.exec_time)

pool.close()
```

+++ {"pycharm": {"name": "#%% md\n"}}

## Processes

```python
import multiprocessing
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


# initialize the process pool and create the runner
n_processes = 4
pool = multiprocessing.Pool(n_processes)
runner = StarmapParallelization(pool.starmap)

# define the problem by passing the starmap interface of the process pool
problem = MyProblem(elementwise_runner=runner)

res = minimize(problem, GA(), termination=("n_gen", 50), seed=1)
print('Processes:', res.exec_time)

pool.close()
```
