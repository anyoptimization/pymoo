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

.. _nb_radar:
```

## Radar Plot


+++

Let us start by generating some data:

```{code-cell} ipython3
import numpy as np

np.random.seed(3)

ideal_point = np.array([0.15, 0.1, 0.2, 0.1, 0.1])
nadir_point = np.array([0.85, 0.9, 0.95, 0.9, 0.85])

F = np.random.random((1, 5)) * (nadir_point - ideal_point) + ideal_point
print(F)
```

If the values should not be normalized, then we can plot the ideal and nadir point.
This keeps the absolute values of each objective. The outer shape represents the nadir point, the inner area the ideal point. All points will lie in the area spanned by those two points additionally.

```{code-cell} ipython3
from pymoo.visualization.radar import Radar

plot = Radar(bounds=[ideal_point, nadir_point], normalize_each_objective=False)
plot.add(F)
plot.show()
```

But if the scale of the objective is too different, then normalization is recommended. Then, the ideal point is just the point in the middle, and the nadir point is now symmetric.

```{code-cell} ipython3
plot = Radar(bounds=[ideal_point, nadir_point])
plot.add(F)
plot.show()
```

```{code-cell} ipython3
F = np.random.random((6, 5)) * (nadir_point - ideal_point) + ideal_point

plot = Radar(bounds=[ideal_point, nadir_point],
             axis_style={"color": 'blue'},
             point_style={"color": 'red', 's': 30})
plot.add(F[:3], color="red", alpha=0.8)
plot.add(F[3:], color="green", alpha=0.8)
plot.show()
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.visualization.radar.Radar
    :noindex:
```
