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

.. _nb_truss2d:
```

# Truss2D

The problem was proposed in <cite data-cite="truss2d"></cite> for the purpose of investigating rules on the Pareto-front.

```{code-cell} ipython3
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

pf = get_problem("truss2d").pareto_front()

sc = Scatter(title="Pareto-front")
sc.add(pf, s=80, facecolors='none', edgecolors='r')
sc.add(pf, plot_type="line", color="black", linewidth=2)
sc.show()
```

```{code-cell} ipython3
sc.reset()
sc.do()
sc.apply(lambda ax: ax.set_yscale("log"))
sc.apply(lambda ax: ax.set_xscale("log"))
sc.show()
```
