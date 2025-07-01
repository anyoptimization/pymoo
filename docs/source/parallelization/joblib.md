---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{raw-cell}
---
pycharm:
  name: '#%% raw

    '
raw_mimetype: text/restructuredtext
---
.. _nb_parallelization_joblib:
```

+++ {"pycharm": {"name": "#%% md\n"}}

# Joblib

+++ {"pycharm": {"name": "#%% md\n"}}

The `JoblibParallelization` class in **pymoo** leverages the popular joblib library to provide powerful and flexible parallelization with multiple execution strategies. This built-in feature offers fine-grained control over parallel execution with support for different backends and advanced configuration options.

**Key Features:**
- Multiple backends: threading, multiprocessing, and loky
- Advanced memory mapping for large datasets
- Automatic batch sizing and load balancing
- Timeout and error handling support

## Basic Usage

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
import os
from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization.joblib import JoblibParallelization
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA

class MyProblem(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = (X ** 2).sum()

# Create joblib runner with default settings (uses all cores)
runner = JoblibParallelization()
problem = MyProblem(elementwise_runner=runner)

res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
print(f'Joblib runtime: {res.exec_time:.2f} sec with {os.cpu_count()} cores')
```

+++ {"pycharm": {"name": "#%% md\n"}}

## Configuration Options

The `JoblibParallelization` class offers extensive configuration options:

### Backend Selection

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
# Use threading backend (good for NumPy operations)
runner_threading = JoblibParallelization(backend="threading")

# Use multiprocessing backend (good for pure Python code)
runner_multiprocessing = JoblibParallelization(backend="multiprocessing")

# Use loky backend (default, robust process-based backend)
runner_loky = JoblibParallelization(backend="loky")
```

### Controlling Number of Jobs

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
# Use all available cores
runner = JoblibParallelization(n_jobs=-1)

# Use specific number of cores
runner = JoblibParallelization(n_jobs=4)

# Leave one core free
runner = JoblibParallelization(n_jobs=-2)
```

### Advanced Options

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
# Configure with timeout and verbosity
runner = JoblibParallelization(
    n_jobs=4,
    backend="loky",
    timeout=10.0,  # 10 second timeout per task
    verbose=10,    # Show progress information
    batch_size="auto"  # Automatic batch sizing
)

# Use generator for streaming results
runner = JoblibParallelization(
    return_as="generator",  # Get results as they complete
    pre_dispatch="2*n_jobs"  # Control task pre-dispatching
)
```

+++ {"pycharm": {"name": "#%% md\n"}}

## Example with Complex Configuration

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
# Advanced configuration for production use
runner = JoblibParallelization(
    n_jobs=-1,                # Use all cores
    backend="loky",           # Robust process-based backend
    timeout=30.0,             # 30 second timeout
    batch_size="auto",        # Automatic batching
    verbose=0,                # No progress output
    max_nbytes="100M",        # Memory map large arrays
    pre_dispatch="2*n_jobs",  # Optimal pre-dispatching
    return_as="list"          # Wait for all results
)

problem = MyProblem(elementwise_runner=runner)
res = minimize(problem, GA(), termination=("n_gen", 100), seed=1)
print(f'Advanced config runtime: {res.exec_time:.2f} sec')
```
