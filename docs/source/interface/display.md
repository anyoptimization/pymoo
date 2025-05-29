---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

+++ {"raw_mimetype": "text/restructuredtext"}

.. _nb_custom_output:

+++

## Display

+++

When running the code, you might have seen some printouts (when using `verbose=True`), which might or might not have made a lot of sense to you. Below, a quick summary of possible printouts you might encounter is provided.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. csv-table:: Types of Output
   :header: "Name", "Description"
   :widths: 5, 30

   **n_gen**, "The current number of generations or iterations until this point."
   **n_eval**, "The number of function evaluations so far."
   **n_nds**, "For multi-objective problems, the number of *non-dominated* solutions of the optima found."
   **cv (min)**,  "The minimum constraint violation (CV) in the current population"
   **cv (avg)**,  "The average constraint violation (CV) of the current population"
   **f_opt**,  "For single-objective problems, the best function value found so far."
   **f_gap**,  "For single-objective problems, the best gap to the optimum (only printed if the optimum is *known*)."
   **eps/indicator**, "For multi-objective problems, the change of the indicator (ideal, nadir, f) over the last few generations (only printed if the Pareto-front is *unknown*). For more information we encourage you to have a look at the corresponding publication (:cite:`running`, `pdf <https://www.egr.msu.edu/~kdeb/papers/c2020003.pdf>`_)."
   **igd/gd/hv**, "For multi-objective problems, the performance indicator (only printed if the Pareto-front is *known*)."
```

The default printouts can vary from algorithm to algorithm and from problem to problem. The type of printout is based on an implementation of the `Display` object. If you like to customize the output, you can also write your own, as shown below:

```{code-cell} ipython3
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output


class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.x_mean = Column("x_mean", width=13)
        self.x_std = Column("x_std", width=13)
        self.columns += [self.x_mean, self.x_std]

    def update(self, algorithm):
        super().update(algorithm)
        self.x_mean.set(np.mean(algorithm.pop.get("X")))
        self.x_std.set(np.std(algorithm.pop.get("X")))


problem = get_problem("zdt2")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 10),
               seed=1,
               output=MyOutput(),
               verbose=True)
```

You have to inherit your custom display `MyOutput` from the `Output` class for your own printout logic.
The `_do` function will be called in each iteration, and the `Problem`, `Evaluator` and `Algorithm` are provided to you. For each column, you can add an entry to `self.output`, which will be formatted and then printed.
