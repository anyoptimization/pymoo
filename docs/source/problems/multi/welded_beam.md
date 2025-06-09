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

.. _nb_welded_beam:
```

# Welded Beam

```{code-cell} ipython3
from pymoo.problems import get_problem
from pymoo.visualization.util import plot

problem = get_problem("welded_beam")
plot(problem.pareto_front(), no_fill=True)
```
