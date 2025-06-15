---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

+++ {"raw_mimetype": "text/restructuredtext"}

.. _nb_interface_algorithm:

+++

## Algorithm

+++

After having defined the problem, a suitable algorithm for optimizing it has to be found. This can be challenging and might require some literature research. *pymoo* offers quite a few standard implementations of well-known algorithms that can be quite useful in obtaining quick results or prototyping.

+++

Each algorithm has different parameters to be initialized. For the functional interface, the `algorithm` object needs to be passed to the `minimize` method, starting the optimization run. For instance, for `NSGA2` the object can be initialized as follows:

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2

algorithm = NSGA2()
```

For more details about algorithms, please have a look at this [tutorial](../algorithms/index.ipynb).
