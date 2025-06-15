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

.. _nb_star:
```

## Star Coordinate Plot

+++

A Star Coordinate Plot maps a higher dimensional space with a non-linear function to two dimensions. Compared to Radviz, points can be outside of the circle.

Let us visualize some test data:

```{code-cell} ipython3
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

ref_dirs = get_reference_directions("uniform", 6, n_partitions=5)
F = get_problem("dtlz1").pareto_front(ref_dirs)
```

A simple Star Coordinate Plot with points can be created by:

```{code-cell} ipython3
from pymoo.visualization.star_coordinate import StarCoordinate

StarCoordinate().add(F).show()
```

```{raw-cell}
The plot can be further customized by supplying a `title`, `label`, and by using other plotting directives from matplotlib. 
```

```{code-cell} ipython3
plot = StarCoordinate(title="Optimization",
                      legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
                      labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"],
                      axis_style={"color": "blue", 'alpha': 0.7},
                      arrow_style={"head_length": 0.015, "head_width": 0.03})
plot.add(F, color="grey", s=20)
plot.add(F[65], color="red", s=70, label="Solution A")
plot.add(F[72], color="green", s=70, label="Solution B")
plot.show()
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.visualization.star_coordinate.StarCoordinate
    :noindex:
```
