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

.. _nb_radviz:
```

## Radviz

+++

Radviz maps a higher-dimensional space with a non-linear function to two dimensions.
Let us visualize some test data:

```{code-cell} ipython3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem


ref_dirs = get_reference_directions("uniform", 6, n_partitions=5)
F = get_problem("dtlz1").pareto_front(ref_dirs)
```

A simple Radviz plot with points can be created by:

```{code-cell} ipython3
from pymoo.visualization.radviz import Radviz
Radviz().add(F).show()
```

The plot can be further customized by supplying a title, labels, and by using the plotting directives from matplotlib. 

```{code-cell} ipython3
plot = Radviz(title="Optimization",
              legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
              labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"],
              endpoint_style={"s": 70, "color": "green"})
plot.set_axis_style(color="black", alpha=1.0)
plot.add(F, color="grey", s=20)
plot.add(F[65], color="red", s=70, label="Solution A")
plot.add(F[72], color="blue", s=70, label="Solution B")
plot.show()
```

Note that radviz plots are by default normalized.

+++

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.visualization.radviz.Radviz
    :noindex:
```
