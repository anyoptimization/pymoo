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

.. _nb_algorithms_usage:
```

# Usage

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Overview
    :class: myOwnStyle
    
    - :ref:`Functional<nb_algorithms_func>` 
    - Object-Oriented Using :ref:`Next<nb_algorithms_next>` 
    - Object-Oriented Using :ref:`Ask And Tell<nb_algorithms_ask_and_tell>`
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_algorithms_func:
```

## Functional

+++

As you might be used to executing algorithms from other frameworks, pymoo offers a functional interface. It requires to pass the problem to be solved, the algorithm to be used, and optionally (but for most algorithms recommended) a termination condition. Other important arguments are discussed in the [Interface](../interface/index.ipynb) tutorial. For executing custom code in between iterations the [Callback](../interface/callback.ipynb) object can be useful. Moreover, it is worth noting that the algorithm object is cloned before being modified. Thus, two calls with the same algorithm object and random seed lead to the same result.

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 10),
               seed=1,
               verbose=True)

# calculate a hash to show that all executions end with the same result
print("hash", res.F.sum())
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_algorithms_object:
```

## Object-oriented

+++

Instead of passing the algorithm to the `minimize` function, it can be used directly for optimization. The first way using the `next` function is available for all algorithms in pymoo. The second way provides a convenient **Ask and Tell** interface, available for most evolutionary algorithms. The reason to use one or the other interface is to have more control during an algorithm's execution or even modify the algorithm object while injecting new solutions.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_algorithms_next:
```

### Next Function

+++

Directly using the `algorithm` object will modify its state during runtime. This allows to ask the object if one more iteration shall be executed or not by calling `algorithm.has_next()`. As soon as the termination criterion has been satisfied, this will return `False`, ending the run. 
Here, we show a custom printout in each iteration (from the second iteration on). Of course, more sophisticated procedures can be incorporated.

```{code-cell} ipython3
import datetime

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100)

# prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
algorithm.setup(problem, termination=('n_gen', 10), seed=1, verbose=False)

# until the algorithm has no terminated
while algorithm.has_next():
    
    # do the next iteration
    algorithm.next()
    
    # do same more things, printing, logging, storing or even modifying the algorithm object
    print(algorithm.n_gen, algorithm.evaluator.n_eval)
    
    
# obtain the result objective from the algorithm
res = algorithm.result()

# calculate a hash to show that all executions end with the same result
print("hash", res.F.sum())
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_algorithms_ask_and_tell:
```

### Ask and Tell

+++

The `next` method already provides much more control over the algorithm executing than the functional interface. However, the call of the `next` function on the algorithm object still is considered a black box. This is where the **Ask and Tell** interface comes into play. Instead of calling one function, two function calls are executed. First, `algorithm.ask()` returns a solution set to be evaluated, and second, `algorithm.tell(solutions)` receives the evaluated solutions to proceed to the next generation. This gives even further control over the run. 

+++

#### Problem-Depdendent

A possible implementation of using this interface can look as follows:

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100)

# prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
algorithm.setup(problem, termination=('n_gen', 10), seed=1, verbose=False)

# until the algorithm has no terminated
while algorithm.has_next():

    # ask the algorithm for the next solution to be evaluated
    pop = algorithm.ask()

    # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
    algorithm.evaluator.eval(problem, pop)

    # returned the evaluated individuals which have been evaluated or even modified
    algorithm.tell(infills=pop)

    # do same more things, printing, logging, storing or even modifying the algorithm object
    print(algorithm.n_gen, algorithm.evaluator.n_eval)

# obtain the result objective from the algorithm
res = algorithm.result()

# calculate a hash to show that all executions end with the same result
print("hash", res.F.sum())
```

#### Problem-independent

Since the evaluation is directly the step between the ask-and-tell calls, the evaluation function of the problem (`_evaluate`) is not even necessary anymore and the evaluation can be moved into the for-loop. We refer to this as the problem-independent execution. However, even in this case, some meta-data about the problem (number of variables, objectives, bounds) need to be provided.

```{code-cell} ipython3
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

problem = Problem(n_var=30, n_obj=2, n_constr=0, xl=np.zeros(30), xu=np.ones(30))

# create the algorithm object
algorithm = NSGA2(pop_size=100)

# let the algorithm object never terminate and let the loop control it
termination = NoTermination()

# create an algorithm object that never terminates
algorithm.setup(problem, termination=termination)

# fix the random seed manually
np.random.seed(1)

# until the algorithm has no terminated
for n_gen in range(10):
    # ask the algorithm for the next solution to be evaluated
    pop = algorithm.ask()

    # get the design space values of the algorithm
    X = pop.get("X")

    # implement your evaluation. here ZDT1
    f1 = X[:, 0]
    v = 1 + 9.0 / (problem.n_var - 1) * np.sum(X[:, 1:], axis=1)
    f2 = v * (1 - np.power((f1 / v), 0.5))
    F = np.column_stack([f1, f2])

    static = StaticProblem(problem, F=F)
    Evaluator().eval(static, pop)

    # returned the evaluated individuals which have been evaluated or even modified
    algorithm.tell(infills=pop)

    # do same more things, printing, logging, storing or even modifying the algorithm object
    print(algorithm.n_gen)

# obtain the result objective from the algorithm
res = algorithm.result()

# calculate a hash to show that all executions end with the same result
print("hash", res.F.sum())
```
