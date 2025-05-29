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

.. _nb_algorithms_init:
```

# Initialization

+++

Algorithms are directly initialized using the corresponding constructor.

+++

Directly initializing the object keeps the code clean and if you use an IDE lets you quickly jump to the definition of the algorithm and find hyperparameters to modify.

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
algorithm = NSGA2()
```
