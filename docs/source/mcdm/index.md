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

.. _nb_decision_making:
```

.. meta::
   :description: Multi-criteria Decision Making addresses the selection of a solution set with multiple conflicting objectives.

+++

.. meta::
   :keywords: Multi-criteria Decision Making, MCDM, Multi-objective Optimization, Python

+++

## Multi-Criteria Decision Making (MCDM)

+++

The focus of pymoo is on optimization methods itself. However, some basic multi-criteria decision making methods are available:

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_compromise:
```

### Compromise Programming

+++

We can use any scalarization method and use it for post-processing. Let us assume our algorithm has converged to the Pareto-front:

```{code-cell} ipython3
from pymoo.problems import get_problem

F = get_problem("zdt1").pareto_front() 
```

Then, we initialize weights and our decomposition function:

```{code-cell} ipython3
import numpy as np

from pymoo.decomposition.asf import ASF

weights = np.array([0.5, 0.5])
decomp = ASF()
```

We apply the decomposition and retrieve the best value (here minimum):

```{code-cell} ipython3
I = decomp(F, weights).argmin()
print("Best regarding decomposition: Point %s - %s" % (I, F[I]))
```

Visualize it:

```{code-cell} ipython3
import numpy as np
from pymoo.visualization.scatter import Scatter

F = np.array(F)
plot = Scatter()
plot.add(F, color="blue", alpha=0.2, s=10)
plot.add(np.array([F[I]]), color="red", s=30)
plot.do()
plot.apply(lambda ax: ax.arrow(0, 0, *weights, color='black', 
                               head_width=0.01, head_length=0.01, alpha=0.4))
plot.show()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_pseudo_weights:
```

### Pseudo-Weights

+++

A simple way to choose a solution out of a solution set in the context of multi-objective optimization is the pseudo-weight vector approach proposed in <cite data-cite="multi_objective_book"></cite>. Respectively, the pseudo weight $w_i$ for the i-ith objective function can be calculated by:

\begin{equation}
w_i = \frac{(f_i^{max} - f_i {(x)}) \, /\,  (f_i^{max} - f_i^{min})}{\sum_{m=1}^M (f_m^{max} - f_m (x)) \, /\,  (f_m^{max} - f_m^{min})}  
\end{equation}

This equation calculates the normalized distance to the worst solution regarding each objective $i$. Please note that for non-convex Pareto fronts the pseudo weight does not correspond to the result of an optimization using the weighted sum. However, for convex Pareto-fronts the pseudo weights are an indicator of the location in the objective space.

```{code-cell} ipython3
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.petal import Petal

ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)
F = get_problem("dtlz1").pareto_front(ref_dirs)
F = np.array(F)

weights = np.array([0.25, 0.25, 0.25, 0.25])
a, pseudo_weights = PseudoWeights(weights).do(F, return_pseudo_weights=True)

weights = np.array([0.4, 0.20, 0.15, 0.25])
b, pseudo_weights = PseudoWeights(weights).do(F, return_pseudo_weights=True)

plot = Petal(bounds=(0, 0.5), reverse=True)
plot.add(F[[a, b]])
plot.show()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_high_tradeoff:
```

### High Trade-off Points

+++

Furthermore, high trade-off points are usually of interest. We have implemented the trade-off metric proposed in <cite data-cite="high-tradeoff"></cite>. An example for 2 and 3 dimensional solution is given below:

```{code-cell} ipython3
import os

import numpy as np
from pymoo.visualization.scatter import Scatter
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints

pf = np.loadtxt("knee-2d.out")
dm = HighTradeoffPoints()

I = dm(pf)

plot = Scatter()
plot.add(pf, alpha=0.2)
plot.add(pf[I], color="red", s=100)
plot.show()
```

```{code-cell} ipython3
pf = np.loadtxt("knee-3d.out")

I = dm(pf)

plot = Scatter(angle=(10, 140))
plot.add(pf, alpha=0.2)
plot.add(pf[I], color="red", s=100)
plot.show()
```
