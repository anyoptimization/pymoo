---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

+++ {"raw_mimetype": "text/restructuredtext"}

.. _nb_callback:

+++

## Callback

+++

A `Callback` class can be used to receive a notification of the algorithm object each generation.
This can be useful to track metrics, do additional calculations, or even modify the algorithm object during the run.
The latter is only recommended for experienced users.

The example below implements a less memory-intense version of keeping track of the convergence. A posteriori analysis can on the one hand, be done by using the `save_history=True` option. This, however, stores a deep copy of the `Algorithm` object in each iteration. This might be more information than necessary, and thus, the `Callback` allows to select only the information necessary to be analyzed when the run has terminated. Another good use case can be to visualize data in each iteration in real-time.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Tip
    :class: myOwnStyle

    The callback has **full** access to the algorithm object and thus can also alter it. However, the callback's purpose is not to customize an algorithm but to store or process data.
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())


problem = get_problem("sphere")

algorithm = GA(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 20),
               seed=1,
               callback=MyCallback(),
               verbose=True)

val = res.algorithm.callback.data["best"]
plt.plot(np.arange(len(val)), val)
plt.show()

```

**Note** that the `Callback` object from the `Result` object needs to be accessed via `res.algorithm.callback` because the original object keeps unmodified to ensure reproducibility.

+++

For completeness, the history-based convergence analysis looks as follows:

```{code-cell} ipython3
res = minimize(problem,
               algorithm,
               ('n_gen', 20),
               seed=1,
               save_history=True)

val = [e.opt.get("F")[0] for e in res.history]
plt.plot(np.arange(len(val)), val)
plt.show()
```
