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

.. _nb_sampling:
```

## Sampling

+++

In the beginning, initial points need to be sampled. *pymoo* offers different sampling methods depending on the variable types.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_sampling_random:
```

### Random Sampling

```{code-cell} ipython3
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.visualization.util import plot

problem = Problem(n_var=2, xl=0, xu=1)

sampling = FloatRandomSampling()

X = sampling(problem, 200).get("X")
plot(X, no_fill=True)
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_sampling_lhs:
```

### Latin Hypercube Sampling

```{code-cell} ipython3
from pymoo.operators.sampling.lhs import LHS

sampling = LHS()

X = sampling(problem, 200).get("X")
plot(X, no_fill=True)
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autofunction:: pymoo.core.sampling.Sampling
    :noindex:
```
