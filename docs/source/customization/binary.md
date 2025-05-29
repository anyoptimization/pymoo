---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

+++ {"raw_mimetype": "text/restructuredtext"}

.. _nb_binary:

+++

## Binary Variable Problem

Mostly, *pymoo* was made for continuous problems, but of course, other variable types can be used as well. The genetic algorithm is a modular class. By modifying the sampling, crossover, and mutation (in some cases also repair), different kinds of variable types can be used (also more complicated ones such as trees, graphs, ...)

+++

In the following, the classical knapsack problem is considered. A single-objective genetic algorithm with a random initial population, half uniform binary crossover and a bitflip mutation is initialized.

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.single.knapsack import create_random_knapsack_problem

problem = create_random_knapsack_problem(30)

algorithm = GA(
    pop_size=200,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(),
    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               verbose=False)

print("Best solution found: %s" % res.X.astype(int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)

```
