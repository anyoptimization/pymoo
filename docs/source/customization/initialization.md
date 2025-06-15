---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

+++ {"raw_mimetype": "text/restructuredtext"}

.. _nb_initialization:

+++

# Biased Initialization



+++

One way of customizing an algorithm is a biased initial population. This can be very helpful if expert knowledge already exists, and known solutions should be improved. In the following, two different ways of initialization are provided: **a)** just providing the design space of the variables and **b)** a `Population` object where the objectives and constraints are provided and do not need to be calculated again.

**NOTE:** This works with all **population-based** algorithms in *pymoo*. Technically speaking, all algorithms which inherit from `GeneticAlgorithm`. For **local-search** based algorithm, the initial solution can be provided by setting `x0` instead of `sampling`.

+++

## By Array

```{code-cell} ipython3
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt2")

X = np.random.random((300, problem.n_var))

algorithm = NSGA2(pop_size=100, sampling=X)

minimize(problem,
         algorithm,
         ('n_gen', 10),
         seed=1,
         verbose=True)
```

## By Population (pre-evaluated)

```{code-cell} ipython3
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.optimize import minimize

problem = get_problem("zdt2")

# create initial data and set to the population object
X = np.random.random((300, problem.n_var))
pop = Population.new("X", X)
Evaluator().eval(problem, pop)

algorithm = NSGA2(pop_size=100, sampling=pop)

minimize(problem,
         algorithm,
         ('n_gen', 10),
         seed=1,
         verbose=True)
```
