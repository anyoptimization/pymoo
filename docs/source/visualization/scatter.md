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

.. _nb_scatter:
```

## Scatter Plot

+++

The traditional scatter plot is mostly used for lower dimensional objective spaces.

+++

### Scatter 2D

```{code-cell} ipython3
from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem

F = get_problem("zdt3").pareto_front()
Scatter().add(F).show()
```

```{raw-cell}
The plot can be further customized by supplying a title, labels, and by using the plotting directives from matplotlib. 
```

```{code-cell} ipython3
F = get_problem("zdt3").pareto_front(use_cache=False, flatten=False)
plot = Scatter()
plot.add(F, s=30, facecolors='none', edgecolors='r')
plot.add(F, plot_type="line", color="black", linewidth=2)
plot.show()
```

### Scatter 3D

```{code-cell} ipython3
from pymoo.util.ref_dirs import get_reference_directions

ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)

F = get_problem("dtlz1").pareto_front(ref_dirs)

plot = Scatter()
plot.add(F)
plot.show()
```

### Scatter ND / Pairwise Scatter Plots

```{code-cell} ipython3
import numpy as np
F = np.random.random((30, 4))

plot = Scatter(tight_layout=True)
plot.add(F, s=10)
plot.add(F[10], s=30, color="red")
plot.show()
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.visualization.scatter.Scatter
    :noindex:
```
