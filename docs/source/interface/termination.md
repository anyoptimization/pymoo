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
:raw_mimetype: text/restructuredtext

.. _nb_interface_termination:
```

## Termination Criterion

+++

Whenever an algorithm is executed, it needs to be decided in each iteration whether the optimization run shall be continued or not.
Many different ways exist of how to determine when a run of an algorithm should be terminated. Next, termination criteria specifically developed for single or multi-objective optimization as well as more generalized, for instance, limiting the number of iterations of an algorithm, are described.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Tip
    :class: myOwnStyle

    The termination of your optimization procedure is **important**. Running the algorithm not long enough can lead to unsatisfactory results; however, running it too long might waste function evaluations and thus computational resources.
```

### Default Termination ('default')

+++

We have recently developed a termination criterion set if no termination is supplied to the `minimize()` method:

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt1")
algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               seed=1)

print(res.algorithm.n_gen)
```

This allows you to terminate based on a couple of criteria also explained later on this page. 
Commonly used are the movement in the design space `xtol` and the convergence in the constraint space `cvtol` and objective space `ftol`.
To provide an upper bound for the algorithm, we recommend supplying a maximum number of generations `n_max_gen` or function evaluations `n_max_evals`.

Moreover, it is worth mentioning that tolerance termination is based on a sliding window. Not only the last generation, but a sequence of the last `period` generations are used to calculate and compare the tolerances with a bound defined by the user.

**Parameter Naming Note**: PyMoo uses consistent parameter naming - `xtol` (design space tolerance), `ftol` (objective space tolerance), `cvtol` (constraint violation tolerance), `period` (number of generations in sliding window), and `n_max_gen` (maximum generations).

+++

By default for multi-objective problems, the termination will be set to

```{code-cell} ipython3
from pymoo.termination.default import DefaultMultiObjectiveTermination

termination = DefaultMultiObjectiveTermination(
    xtol=1e-8, 
    cvtol=1e-6, 
    ftol=0.0025, 
    period=30,
    n_max_gen=1000,
    n_max_evals=100000
)
```

And for single-objective optimization to

```{code-cell} ipython3
from pymoo.termination.default import DefaultSingleObjectiveTermination

termination = DefaultSingleObjectiveTermination(
    xtol=1e-8, 
    cvtol=1e-6, 
    ftol=1e-6, 
    period=20,
    n_max_gen=1000,
    n_max_evals=100000
)
```

### Customizing Termination Criteria

+++

You can customize termination by creating your own termination object or by modifying the default ones. Here are common scenarios:

**Removing specific criteria**: Set unwanted parameters to very large values to effectively disable them:

```{code-cell} ipython3
from pymoo.termination.default import DefaultMultiObjectiveTermination

# Create termination with effectively disabled ftol (objective space tolerance)
termination = DefaultMultiObjectiveTermination(
    xtol=1e-8, 
    cvtol=1e-6, 
    ftol=1.0,  # Very large value effectively disables this criterion
    period=30,
    n_max_gen=100,  # Rely mainly on generation limit
    n_max_evals=10000
)
```

**Combining multiple criteria**: You can combine different termination conditions:

```{code-cell} ipython3
from pymoo.termination.collection import TerminationCollection
from pymoo.termination import get_termination

# Terminate when ANY condition is met
termination = TerminationCollection(
    get_termination("n_gen", 200),      # Max 200 generations
    get_termination("time", "00:05:00") # Max 5 minutes
)
```

**Single termination criterion**: Use only one specific criterion:

```{code-cell} ipython3
# Only use generation-based termination
termination = get_termination("n_gen", 100)

# Only use evaluation-based termination  
termination = get_termination("n_eval", 5000)

# Only use time-based termination
termination = get_termination("time", "00:10:00")
```

### Parameter Reference

+++

For clarity, here's a reference of all termination parameters and their consistent naming:

| Parameter | Description | Type | Example |
|-----------|-------------|------|---------|
| `xtol` | Design space tolerance - stops when decision variables change less than this value (absolute) | float | `1e-8` |
| `ftol` | Objective space tolerance - stops when objective values change less than this value (relative for MOO, absolute for SOO) | float | `1e-6` |
| `cvtol` | Constraint violation tolerance - stops when constraint violations are below this value (absolute) | float | `1e-6` |
| `period` | Number of generations to consider in sliding window for tolerance calculations | int | `30` |
| `n_max_gen` | Maximum number of generations before forced termination | int | `1000` |
| `n_max_evals` | Maximum number of function evaluations before forced termination | int | `100000` |
| `n_skip` | Calculate termination criterion every n_skip generations (for performance) | int | `5` |

**Note**: All tolerance parameters (`xtol`, `ftol`, `cvtol`) can be set to large values (e.g., `1.0`) to effectively disable that specific termination criterion.

### How Termination Criteria Work Together

+++

**Multiple Criteria Logic**: When multiple termination criteria are specified (e.g., `xtol`, `ftol`, `n_max_gen`), the algorithm terminates when **ANY** of the criteria is satisfied (OR logic). This means:

- If `xtol=1e-8` AND `n_max_gen=100` are both specified, the algorithm stops when either the design space changes become smaller than `1e-8` OR when 100 generations are reached, whichever happens first.
- This ensures the algorithm doesn't run indefinitely if tolerance criteria are never met, and also allows early stopping when convergence is detected.

**Absolute vs Relative Tolerances**: 

- **`xtol` (Design Space Tolerance)**: Uses **absolute** tolerance. It measures the absolute change in decision variables between generations. For example, `xtol=1e-8` means the algorithm stops when the maximum absolute change in any decision variable is less than 1e-8.

- **`ftol` (Objective Space Tolerance)**: Uses **relative** tolerance for multi-objective problems and **absolute** tolerance for single-objective problems. For multi-objective optimization, it considers the relative change in the hypervolume or other convergence metrics.

- **`cvtol` (Constraint Violation Tolerance)**: Uses **absolute** tolerance, measuring the absolute constraint violation values.

**Example of Combined Criteria**:

```{code-cell} ipython3
from pymoo.termination.default import DefaultMultiObjectiveTermination

# Algorithm will stop when ANY of these conditions is met:
termination = DefaultMultiObjectiveTermination(
    xtol=1e-8,        # Stop if design variables change < 1e-8 (absolute)
    ftol=0.0025,      # Stop if objective improvement < 0.0025 (relative)
    cvtol=1e-6,       # Stop if constraint violations < 1e-6 (absolute)
    n_max_gen=1000,   # Stop after 1000 generations maximum
    n_max_evals=100000 # Stop after 100000 function evaluations maximum
)
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_n_eval:
```

### Number of Evaluations ('n_eval')

+++

The termination can simply be reached by providing an upper bound for the number of function evaluations. Whenever in an iteration, the number of function evaluations is greater than this upper bound the algorithm terminates.

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize

problem = get_problem("zdt3")
algorithm = NSGA2(pop_size=100)
termination = get_termination("n_eval", 300)

res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=True)
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_n_gen:
```

### Number of Generations ('n_gen')

+++

Moreover, the number of generations / iterations can be limited as well.

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize

problem = get_problem("zdt3")
algorithm = NSGA2(pop_size=100)
termination = get_termination("n_gen", 10)

res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=True)
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_time:
```

### Based on Time ('time')

+++

The termination can also be based on the time of the algorithm to be executed. For instance, to run an algorithm for 3 seconds the termination can be defined by `get_termination("time", "00:00:03")` or for 1 hour and 30 minutes by `get_termination("time", "01:30:00")`.

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize

problem = get_problem("zdt3")
algorithm = NSGA2(pop_size=100)
termination = get_termination("time", "00:00:03")

res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=False)

print(res.algorithm.n_gen)
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_xtol:
```

### Design Space Tolerance ('xtol')

+++

Also, we can track the change in the design space. For a parameter explanation, please have a look at 'ftol'.

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.termination.robust import RobustTermination

problem = get_problem("zdt3")
algorithm = NSGA2(pop_size=100)
termination = RobustTermination(DesignSpaceTermination(tol=0.01), period=20)

res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=False)

print(res.algorithm.n_gen)
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_ftol:
```

### Objective Space Tolerance ('ftol')

The most interesting stopping criterion is to use objective space change to decide whether to terminate the algorithm. Here, we mostly use a simple and efficient procedure to determine whether to stop or not. We aim to improve it further in the future. If somebody is interested in collaborating, please let us know.

The parameters of our implementation are:

**tol**: What is the tolerance in the objective space on average. If the value is below this bound, we terminate.

**period**: To make the criterion more robust, we consider the last `period` generations and take the maximum. This considers the worst case in a sliding window.

**n_max_gen**: As a fallback, the generation number can be used. For some problems, the termination criterion might not be reached; however, an upper bound for generations can be defined to stop in that case.

**n_skip**: Defines how often the termination criterion is calculated (by default, every n_skip generations to reduce computational overhead).

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt3")

algorithm = NSGA2(pop_size=100)

termination = RobustTermination(
    MultiObjectiveSpaceTermination(tol=0.005, n_skip=5), period=20)


res = minimize(problem,
               algorithm,
               termination,
               pf=True,
               seed=1,
               verbose=False)

print("Generations", res.algorithm.n_gen)
plot = Scatter(title="ZDT3")
plot.add(problem.pareto_front(use_cache=False, flatten=False), plot_type="line", color="black")
plot.add(res.F, facecolor="none", edgecolor="red", alpha=0.8, s=20)
plot.show()
```
