---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

```{raw-cell}
---
pycharm:
  name: '#%% raw

    '
raw_mimetype: text/restructuredtext
---
.. _nb_parallelization:
```

+++ {"pycharm": {"name": "#%% md\n"}}

# Parallelization

+++ {"pycharm": {"name": "#%% md\n"}}

In practice, parallelization is essential and can significantly speed up optimization. 
For population-based algorithms, the evaluation of a set of solutions can be parallelized easily 
by parallelizing the evaluation itself.

+++ {"pycharm": {"name": "#%% md\n"}}

## Vectorized Matrix Operations

One way is using the `NumPy` matrix operations, which has been used for almost all test problems implemented in *pymoo*.
By default, `elementwise_evaluation` is set to `False`, which implies the `_evaluate` retrieves a set of solutions.
Thus, `x` is a matrix where each row is an individual, and each column a variable.

```{code-cell}
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

```{code-cell}
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

+++ {"pycharm": {"name": "#%% md\n"}}

## Starmap Interface

In general, **pymoo** allows passing a `starmap` object to be used for parallelization. 
The `starmap` interface is defined in the Python standard library `multiprocessing.Pool.starmap` [function](https://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#multiprocessing.pool.Pool.starmap).
This allows for excellent and flexible parallelization opportunities. 

**IMPORTANT:** Please note that the problem needs to have set `elementwise_evaluation=True`, which indicates one call of `_evaluate` only takes care of a single solution.


```{code-cell}
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

### Threads

```{code-cell}
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


# initialize the thread pool and create the runner
n_threads = 4
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

# define the problem by passing the starmap interface of the thread pool
problem = MyProblem(elementwise_runner=runner)

res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
print('Threads:', res.exec_time)

pool.close()
```

+++ {"pycharm": {"name": "#%% md\n"}}

### Processes

```{code-cell}
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
import multiprocessing
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


# initialize the thread pool and create the runner
n_proccess = 8
pool = multiprocessing.Pool(n_proccess)
runner = StarmapParallelization(pool.starmap)

# define the problem by passing the starmap interface of the thread pool
problem = MyProblem(elementwise_runner=runner)

res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
print('Threads:', res.exec_time)

pool.close()
```

+++ {"pycharm": {"name": "#%% md\n"}}

**Note:** Here clearly the overhead of serializing and transferring the data are visible.

+++ {"pycharm": {"name": "#%% md\n"}}

## Dask

+++ {"pycharm": {"name": "#%% md\n"}}

More advanced is to distribute the evaluation function to a couple of workers. There exists a couple of frameworks that support the distribution of code. For our framework, we recommend using [Dask](https://dask.org).

Documentation to setup the cluster is available [here](https://docs.dask.org/en/latest/setup/cli.html). You first start a scheduler somewhere and then connect workers to it. Then, a client object connects to the scheduler and distributes the jobs automatically for you.

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import DaskParallelization

from dask.distributed import Client
client = Client()
client.restart()
print("DASK STARTED")

# initialize the thread pool and create the runner
runner = DaskParallelization(client)

# define the problem by passing the starmap interface of the thread pool
problem = MyProblem(elementwise_runner=runner)

res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
print('Threads:', res.exec_time)

client.close()
print("DASK SHUTDOWN")
```

+++ {"pycharm": {"name": "#%% md\n"}}

**Note:** Here, the overhead of transferring data to the workers of Dask is dominating. However, if your problem is computationally more expensive, this shall not be the case anymore.

+++ {"pycharm": {"name": "#%% md\n"}}

## Custom Parallelization

+++ {"pycharm": {"name": "#%% md\n"}}

If you need more control over the parallelization process, we like to provide an example of fully customizable parallelization. The `_evaluate` function gets the whole set of solutions to be evaluated because, by default, `elementwise_evaluation` is disabled.

+++ {"pycharm": {"name": "#%% md\n"}}

### Threads

+++ {"pycharm": {"name": "#%% md\n"}}

Thus, a thread pool can be initialized in the constructor of the `Problem` class and then be used to speed up the evaluation.
The code below basically does what internally happens using the `starmap` interface of *pymoo* directly (with an inline function definition and without some overhead, this is why it is slightly faster).

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
from pymoo.core.problem import Problem

pool = ThreadPool(8)

class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)
        
    def _evaluate(self, X, out, *args, **kwargs):
        
        # define the function
        def my_eval(x):
            return (x ** 2).sum()
            
        # prepare the parameters for the pool
        params = [[X[k]] for k in range(len(X))]

        # calculate the function values in a parallelized manner and wait until done
        F = pool.starmap(my_eval, params)
        
        # store the function values and return them.
        out["F"] = np.array(F)
        
problem = MyProblem()       
```

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
print('Threads:', res.exec_time)
```

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
pool.close()
```

+++ {"pycharm": {"name": "#%% md\n"}}

### Dask

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
import numpy as np
from dask.distributed import Client

from pymoo.core.problem import Problem
from pymoo.optimize import minimize

client = Client(processes=False)

class MyProblem(Problem):

    def __init__(self, *args, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5,
                         elementwise_evaluation=False, *args, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        def fun(x):
            return np.sum(x ** 2)

        jobs = [client.submit(fun, x) for x in X]
        out["F"] = np.row_stack([job.result() for job in jobs])
```

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
problem = MyProblem()

res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
print('Dask:', res.exec_time)

client.close()
```
