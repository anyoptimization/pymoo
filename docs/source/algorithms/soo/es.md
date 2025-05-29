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

.. _nb_es:
```

.. meta::
   :description: Evolutionary Strategy is a well-known algorithm in evolutionary computation consisting of selection and mutation. The standard version has been proposed for real-valued optimization where a Gaussian mutation is applied, and the selection is based on each individual's fitness value.

+++

.. meta::
   :keywords: Evolutionary Strategy, ES,  Multi-modal Optimization, Real-Valued Optimization, Single-objective Optimization, Python

+++

# ES: Evolutionary Strategy

+++

Evolutionary Strategy is a well-known algorithm in evolutionary computation consisting of selection and mutation. The standard version has been proposed for **real-valued** optimization where a gaussian mutation is applied, and the selection is based on each individual's fitness value.

In this implementation, the 1/7 rule creates seven times more offspring than individuals in the current population. The $sigma$ values for the mutation are based on a meta-evolution of surviving individuals.

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("ackley", n_var=10)

algorithm = ES(n_offsprings=200, rule=1.0 / 7.0)

res = minimize(problem,
               algorithm,
               ("n_gen", 200),
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.soo.nonconvex.es.ES
    :noindex:
```
