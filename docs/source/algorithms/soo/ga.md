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

.. _nb_ga:
```

# GA: Genetic Algorithm

This class represents a basic ($\mu+\lambda$) genetic algorithm for single-objective problems. The figure below shows the flow of a genetic algorithm in general. In the following, it is explained how *pymoo* can be customized.

1) **Initial Population:** A starting population is sampled in the beginning. In this framework, this can be either a [Sampling](../../operators/sampling.ipynb) object, which defines different initial sampling strategies, or `Population` where the `X` and `F` values are set, or a simple NumPy array (pop_size x n_var).



2) **Evaluation:** It is executed using the problem defined to be solved.

3) **Survival:** It is often the core of the genetic algorithm used. For a simple single-objective genetic algorithm, the individuals can be sorted by their fitness, and survival of the fittest can be applied.

4) [Selection](../../operators/selection.ipynb): At the beginning of the recombination process, individuals need to be selected to participate in mating. Depending on the crossover, a different number of parents need to be selected. Different kinds of selections can increase the convergence of the algorithm.

5) [Crossover](../../operators/crossover.ipynb): When the parents are selected, the actual mating is done. A crossover operator combines parents into one or several offspring. Commonly, problem information, such as the variable bounds, are needed to perform the mating. For more customized problems, even more, information might be necessary (e.g. current generation, diversity measure of the population, ...)

6) [Mutation](../../operators/mutation.ipynb): It is performed after the offsprings are created through the crossover. Usually, the mutation is executed with a predefined probability. This operator helps to increase the diversity in the population.


+++

<div style="text-align: center;">
    <img src="https://github.com/anyoptimization/pymoo-data/blob/main/docs/images/ga_basic.png?raw=true" width="350">
</div>


+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("g1")

algorithm = GA(
    pop_size=100,
    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autofunction:: pymoo.algorithms.soo.nonconvex.ga.GA
    :noindex:
```
