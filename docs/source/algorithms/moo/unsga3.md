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

.. _nb_unsga3:
```

# U-NSGA-III


The algorithm is implemented based on <cite data-cite="unsga3"></cite>. NSGA-III selects parents randomly for mating. It has been shown that tournament selection performs better than random selection. The *U* stands for *unified* and increases NSGA-III's performance by introducing tournament pressure. 

The mating selection works as follows:

+++

<div style="text-align: center;">
    <img src="https://github.com/anyoptimization/pymoo-data/blob/main/docs/images/unsga3_mating.png?raw=true" width="400">
</div>

+++

### Example

```{code-cell} ipython3
import numpy as np

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("ackley", n_var=30)

# create the reference directions to be used for the optimization - just a single one here
ref_dirs = np.array([[1.0]])

# create the algorithm object
algorithm = UNSGA3(ref_dirs, pop_size=100)

# execute the optimization
res = minimize(problem,
               algorithm,
               termination=('n_gen', 150),
               save_history=True,
               seed=1)

print("UNSGA3: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

U-NSGA-III has for single- and bi-objective problems a tournament pressure which is known to be useful.
In the following, we provide a quick comparison (here just one run, so not a valid experiment) to see the difference in convergence.

```{code-cell} ipython3
_res = minimize(problem,
                NSGA3(ref_dirs, pop_size=100),
                termination=('n_gen', 150),
                save_history=True,
                seed=1)
print("NSGA3: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

ret = [np.min(e.pop.get("F")) for e in res.history]
_ret = [np.min(e.pop.get("F")) for e in _res.history]

plt.plot(np.arange(len(ret)), ret, label="unsga3")
plt.plot(np.arange(len(_ret)), _ret, label="nsga3")
plt.title("Convergence")
plt.xlabel("Generation")
plt.ylabel("F")
plt.legend()
plt.show()
```

+++ {"raw_mimetype": "text/restructuredtext"}

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.unsga3.UNSGA3
    :noindex:
```
