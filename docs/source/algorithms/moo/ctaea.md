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

.. _nb_ctaea:
```

# C-TAEA


This algorithm is implemented based on <cite data-cite="ctaea"></cite> and the authors' [implementation](https://web.archive.org/web/20200916105021/https://cola-laboratory.github.io/docs/publications). The algorithm is based on [Reference Directions](../../misc/reference_directions.ipynb), which need to be provided when initializing the algorithm object.

C-TAEA follows a two archive approach to balance convergence (Convergence Archive CA) and diversity (Diversity Archive DA).

```{code-cell} ipython3
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("c1dtlz1", None, 3, k=5)

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# create the algorithm object
algorithm = CTAEA(ref_dirs=ref_dirs)

# execute the optimization
res = minimize(problem,
               algorithm,
               ('n_gen', 600),
               seed=1,
               verbose=False
               )

sc = Scatter(legend=False, angle=(45, 30))
sc.add(problem.pareto_front(ref_dirs), plot_type='surface', alpha=0.2, label="PF", color="blue")
sc.add(res.F, facecolor="none", edgecolor="red")
sc.show()
```

```{code-cell} ipython3
problem = get_problem("carside")
ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_points=91)
algorithm = CTAEA(ref_dirs=ref_dirs)

res = minimize(problem,
               algorithm,
               ('n_gen', 600),
               seed=1
               )

Scatter().add(res.F, facecolor="none", edgecolor="red").show()
```

## API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.ctaea.CTAEA
    :noindex:
```

Python implementation by [cyrilpic](https://github.com/cyrilpic) based on the [original C code](https://web.archive.org/web/20200916105021/https://cola-laboratory.github.io/docs/publications).
