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

.. _nb_agemoea:
```

.. meta::
   :description: An implementation of AGE-MOEA algorithm to solve many-objective optimization problems without using reference directions. The algorithm estimates the shape of the Pareto front to provide a better way to compute proximity to ideal point and crowding distance.

+++

.. meta::
   :keywords: AGEMOEA, NSGA-II, Non-Dominated Sorting, Multi-objective Optimization, Python

+++

# AGE-MOEA: Adaptive Geometry Estimation based MOEA

+++

AGE-MOEA <cite data-cite="agemoea"></cite> follows the general
outline of [NSGA-II](../moo/nsga2.ipynb) but with a modified crowding distance formula. The non-dominated fronts are sorted using the non-dominated sorting procedure. Then the first front is used for normalization of the objective space and estimation of Pareto front geometry. The `p` parameter of a Minkowski p-norm is estimated using the closest solution from the middle of the first front. The p-norm is then used to compute a survival score that combines distance from the neighbors and proximity to the ideal point.

+++

AGE-MOEA uses a binary tournament mating selection to increase some selection pressure. Each individual is first compared using the rank and then the computed score that represent both proximity and spread.

+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = AGEMOEA(pop_size=100)


res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
```

Moreover, we can customize AGE-MOEA to solve a problem with binary decision variables, for example, ZDT5.

```{code-cell} ipython3
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.problems import get_problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt5")

algorithm = AGEMOEA(pop_size=100,
                    sampling=BinaryRandomSampling(),
                    crossover=TwoPointCrossover(),
                    mutation=BitflipMutation(),
                    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 500),
               seed=1,
               verbose=False)

Scatter().add(res.F, facecolor="none", edgecolor="red").show()
```

This algorithm is based on <cite data-cite="agemoea"></cite> and its Matlab implementation in the PlatEMO library. This Python version has been implemented by [BenCrulis](https://github.com/BenCrulis)  

+++

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.age.AGEMOEA
    :noindex:
```
