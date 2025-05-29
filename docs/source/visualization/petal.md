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

.. _nb_petal:
```

## Petal Diagram

+++

Also, it might be interesting to compare one solution with another. Here, a visual of a single solution regarding its trade-offs.

Let us visualize some test data:

```{code-cell} ipython3
import numpy as np

np.random.seed(1234)
F = np.random.random((1, 6))
print(F)
```

A simple petal plot can be created by:

```{code-cell} ipython3
from pymoo.visualization.petal import Petal

Petal(bounds=[0, 1]).add(F).show()
```

If you prefer to visualize smaller values with a larger area, set `reverse=True`:

```{code-cell} ipython3
Petal(bounds=[0, 1], reverse=True).add(F).show()
```

```{code-cell} ipython3
plot = Petal(bounds=[0, 1],
             cmap="tab20",
             labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"],
             title=("Solution A", {'pad': 20}))
plot.add(F)
plot.show()
```

Each add will plot solutions in a row. Each entry represents a column.
Different solutions can be easily compared.

```{code-cell} ipython3
F = np.random.random((6, 6))
plot = Petal(bounds=[0, 1], title=["Solution %s" % t for t in ["A", "B", "C", "D", "E", "F"]])
plot.add(F[:3])
plot.add(F[3:])
plot.show()
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.visualization.petal.Petal
    :noindex:
```
