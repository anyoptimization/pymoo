---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: default
  display_name: default
  language: python
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_pcp:
```

## Parallel Coordinate Plots

+++

For higher-dimensional data, Parallel Coordinate Plots are a powerful technique to analyze how dense solutions are distributed in different ranges regarding each coordinate.

Let us create some data for visualization:

```{code-cell} ipython3
from pymoo.problems.many.dtlz import DTLZ1
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

ref_dirs = UniformReferenceDirectionFactory(6, n_partitions=5)() * [2, 4, 8, 16, 32, 64]
F = DTLZ1().pareto_front(ref_dirs)
```

This is the Pareto-front for the DTLZ1 test problem with six objectives, with some scale added. We add a different scaling to show the effect of normalization later on. Let us assume our algorithm converged after some generations, and this is the result set.

```{code-cell} ipython3
from pymoo.visualization.pcp import PCP
PCP().add(F).show()
```

This gives an idea of the overall result set. 
Let us assume we identified solution 50 and 75 to be more of interest and would like to highlight them in our plot:

```{code-cell} ipython3
plot = PCP()
plot.set_axis_style(color="grey", alpha=0.5)
plot.add(F, color="grey", alpha=0.3)
plot.add(F[50], linewidth=5, color="red")
plot.add(F[75], linewidth=5, color="blue")
plot.show()
```

Please note that the PCP object just is a wrapper around a matplotlib figure. All options that apply for plotting the corresponding type (here `plot`, but it can also be `scatter`, `polygon`, ...) can be used, such as `linewidth`, `color` or `alpha`.

+++

Some more options to be used in a plot

```{code-cell} ipython3
plot = PCP(title=("Run", {'pad': 30}),
           n_ticks=10,
           legend=(True, {'loc': "upper left"}),
           labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"]
           )

plot.set_axis_style(color="grey", alpha=1)
plot.add(F, color="grey", alpha=0.3)
plot.add(F[50], linewidth=5, color="red", label="Solution A")
plot.add(F[75], linewidth=5, color="blue", label="Solution B")
plot.show()
```

Moreover, if the boundaries should be set manually, this can be achieved by turning the default normalization off and providing them. Either directly as a NumPy array or just an integer to be set for all axes.

```{code-cell} ipython3
plot.reset()
plot.normalize_each_axis = False
plot.bounds = [[1,1,1,2,2,5],[32,32,32,32,32,32]]
plot.show()
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.visualization.pcp.PCP
    :noindex:
```
