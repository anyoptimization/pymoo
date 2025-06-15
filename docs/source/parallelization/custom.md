---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: ''
  name: ''
---

```{raw-cell}
---
pycharm:
  name: '#%% raw

    '
raw_mimetype: text/restructuredtext
---
.. _nb_parallelization_custom:
```

+++ {"pycharm": {"name": "#%% md\n"}}

# Custom Parallelization

+++ {"pycharm": {"name": "#%% md\n"}}

If you need more control over the parallelization process, we like to provide an example of fully customizable parallelization. The `_evaluate` function gets the whole set of solutions to be evaluated because, by default, `elementwise_evaluation` is disabled.

+++ {"pycharm": {"name": "#%% md\n"}}

## Thread Pool Implementation

+++ {"pycharm": {"name": "#%% md\n"}}

Thus, a thread pool can be initialized in the constructor of the `Problem` class and then be used to speed up the evaluation.
The code below basically does what internally happens using the `starmap` interface of *pymoo* directly (with an inline function definition and without some overhead, this is why it is slightly faster).

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
import numpy as np
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

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

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
print('Threads:', res.exec_time)
```

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
pool.close()
```

+++ {"pycharm": {"name": "#%% md\n"}}

## Custom Implementation with Progress Tracking

You can implement custom parallelization with progress tracking:

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

class MyProgressProblem(Problem):
    def __init__(self, n_processes=4, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)
        self.n_processes = n_processes
        
    def _evaluate(self, X, out, *args, **kwargs):
        def eval_single(x):
            # Simulate expensive computation
            import time
            time.sleep(0.01)
            return (x ** 2).sum()
        
        results = [None] * len(X)
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # Submit all jobs
            future_to_idx = {executor.submit(eval_single, X[i]): i 
                           for i in range(len(X))}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_idx), 
                             total=len(X), desc="Evaluating"):
                idx = future_to_idx[future]
                results[idx] = future.result()
        
        out["F"] = np.array(results)
```
