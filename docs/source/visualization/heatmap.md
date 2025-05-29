---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: default
  language: python
  name: default
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_heat:
```

## Heatmap

+++

For getting an idea of the distribution of values, heatmaps can be used.

Let us visualize some test data:

```{code-cell} ipython3
import numpy as np

np.random.seed(1234)
F = np.random.random((4, 6))
```

A simple heatmap can be created by:

```{code-cell} ipython3
from pymoo.visualization.heatmap import Heatmap
Heatmap().add(F).show()
```

```{raw-cell}
By default, larger values are represented with white and smaller values with the corresponding color:
```

```{code-cell} ipython3
Heatmap(bounds=[0,1]).add(np.ones((1, 6))).show() 
```

This behavior can be changed by setting `reverse` to False.

```{code-cell} ipython3
Heatmap(bounds=[0,1],reverse=False).add(np.ones((1, 6))).show() 
```

The plot can be further customized by supplying a title, labels, and by using the plotting directives from matplotlib. Also, colors can be changed:

```{code-cell} ipython3
plot = Heatmap(title=("Optimization", {'pad': 15}),
               cmap="Oranges_r",
               solution_labels=["Solution A", "Solution B", "Solution C", "Solution D"],
               labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"])
plot.add(F)
plot.show()
```

Moreover, the values can be sorted lexicographically by objective(s) - and by default, the selected objective is inserted in position 0 of the range of objectives. Also, boundaries can be changed. Otherwise, it is scaled according to the smallest and largest values supplied.

```{code-cell} ipython3
F = np.random.random((30, 6))

plot = Heatmap(figsize=(10,30),
               bounds=[0,1],
               order_by_objectives=0,
               solution_labels=None,
               labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"],
               cmap="Greens_r")

plot.add(F, aspect=0.2)
plot.show()
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.visualization.heatmap.Heatmap
    :noindex:
```
