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

.. _nb_algorithms_hyperparameters:
```

# Hyperparameters

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Info
    :class: myOwnStyle
    
    Hyperparameter optimization is a new feature available since version **0.6.0**. In general, this is quite a challenging and computationally expensive topic, and only a few basics are presented in this guide. If you are interested in contributing or collaborating, please let us know to enrich this module with more robust and better features.
```

Most algorithms have **hyperparameters**. For some optimization methods the parameters are already defined and can directly be optimized. For instance, for Differential Evolution (DE) the parameters can be found by:

```{code-cell} ipython3
import json
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.parameters import get_params, flatten, set_params, hierarchical

algorithm = DE()
flatten(get_params(algorithm))
```

If not provided directly, when initializing a `HyperparameterProblem` these variables are directly used for optimization.

+++

Secondly, one needs to define what exactly should be optimized. For instance, for a single run on a problem (with a fixed random seed) using the well-known parameter optimization toolkit [Optuna](https://optuna.org), the implementation may look as follows:

```{code-cell} ipython3
from pymoo.algorithms.hyperparameters import SingleObjectiveSingleRun, HyperparameterProblem
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.parameters import set_params, hierarchical
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere

algorithm = G3PCX()

problem = Sphere(n_var=10)
n_evals = 500

performance = SingleObjectiveSingleRun(problem, termination=("n_evals", n_evals), seed=1)

res = minimize(HyperparameterProblem(algorithm, performance),
               Optuna(),
               termination=('n_evals', 50),
               seed=1,
               verbose=False)

hyperparams = res.X
print(hyperparams)
set_params(algorithm, hierarchical(hyperparams))

res = minimize(Sphere(), algorithm, termination=("n_evals", n_evals), seed=1)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

Of course, you can also directly use the `MixedVariableGA` available in our framework:

```{code-cell} ipython3
from pymoo.algorithms.hyperparameters import SingleObjectiveSingleRun, HyperparameterProblem
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.parameters import set_params, hierarchical
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere


algorithm = G3PCX()

problem = Sphere(n_var=10)
n_evals = 500

performance = SingleObjectiveSingleRun(problem, termination=("n_evals", n_evals), seed=1)

res = minimize(HyperparameterProblem(algorithm, performance),
               MixedVariableGA(pop_size=5),
               termination=('n_evals', 50),
               seed=1,
               verbose=False)

hyperparams = res.X
print(hyperparams)
set_params(algorithm, hierarchical(hyperparams))

res = minimize(Sphere(), algorithm, termination=("n_evals", n_evals), seed=1)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

Now, optimizing the parameters for a **single random seed** is often not desirable. And this is precisely what makes hyper-parameter optimization computationally expensive. So instead of using just a single random seed, we can use the `MultiRun` performance assessment to average over multiple runs as follows:

```{code-cell} ipython3
from pymoo.algorithms.hyperparameters import HyperparameterProblem, MultiRun, stats_single_objective_mean
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.parameters import set_params, hierarchical
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere


algorithm = G3PCX()

problem = Sphere(n_var=10)
n_evals = 500
seeds = [5, 50, 500]

performance = MultiRun(problem, seeds=seeds, func_stats=stats_single_objective_mean, termination=("n_evals", n_evals))

res = minimize(HyperparameterProblem(algorithm, performance),
               MixedVariableGA(pop_size=5),
               termination=('n_evals', 50),
               seed=1,
               verbose=True)

hyperparams = res.X
print(hyperparams)
set_params(algorithm, hierarchical(hyperparams))

res = minimize(Sphere(), algorithm, termination=("n_evals", n_evals), seed=5)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

Another way of performance measure is the number of evaluations until a specific goal has been reached. For single-objective optimization, such a goal is most likely until a minimum function value has been found. Thus, for the termination, we use `MinimumFunctionValueTermination` with a value of `1e-5`. We run the method for each random seed until this value has been reached or at most `500` function evaluations have taken place. The performance is then measured by the average number of function evaluations (`func_stats=stats_avg_nevals`) to reach the goal.

```{code-cell} ipython3
from pymoo.algorithms.hyperparameters import HyperparameterProblem, MultiRun, stats_avg_nevals
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.parameters import set_params, hierarchical
from pymoo.core.termination import TerminateIfAny
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere
from pymoo.termination.fmin import MinimumFunctionValueTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination

algorithm = G3PCX()

problem = Sphere(n_var=10)

termination = TerminateIfAny(MinimumFunctionValueTermination(1e-5), MaximumFunctionCallTermination(500))

performance = MultiRun(problem, seeds=[5, 50, 500], func_stats=stats_avg_nevals, termination=termination)

res = minimize(HyperparameterProblem(algorithm, performance),
               MixedVariableGA(pop_size=5),
               ('n_evals', 50),
               seed=1,
               verbose=True)

hyperparams = res.X
print(hyperparams)
set_params(algorithm, hierarchical(hyperparams))

res = minimize(Sphere(), algorithm, termination=("n_evals", res.f), seed=5)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```
