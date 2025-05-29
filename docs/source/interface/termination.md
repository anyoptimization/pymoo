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
Commonly used are the movement in the design space `f_tol` and the convergence in the constraint `cv_tol` and objective space `f_tol`.
To provide an upper bound for the algorithm, we recommend supplying a maximum number of generations `n_max_gen` or function evaluations `n_max_evals`.

Moreover, it is worth mentioning that tolerance termination is based on a sliding window. Not only the last, but a sequence of the `period` generations are used to calculate and compare the tolerances with a bound defined by the user.

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

```{raw-cell}
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
.. _nb_ftol:
```

### Objective Space Tolerance ('ftol')

The most interesting stopping criterion is to use objective space change to decide whether to terminate the algorithm. Here, we mostly use a simple and efficient procedure to determine whether to stop or not. We aim to improve it further in the future. If somebody is interested in collaborating, please let us know.

The parameters of our implementation are:

**tol**: What is the tolerance in the objective space on average. If the value is below this bound, we terminate.

**n_last**: To make the criterion more robust, we consider the last $n$ generations and take the maximum. This considers the worst case in a window.

**n_max_gen**: As a fallback, the generation number can be used. For some problems, the termination criterion might not be reached; however, an upper bound for generations can be defined to stop in that case.

**nth_gen**: Defines whenever the termination criterion is calculated by default, every 10th generation.

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
