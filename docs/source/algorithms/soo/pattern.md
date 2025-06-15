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

.. _nb_pattern_search:
```

# Pattern Search

+++

An implementation of the well-known Hooke and Jeeves Pattern Search <cite data-cite="pattern_search"></cite> for single-objective optimization which makes use of *exploration* and *pattern* moves in an alternating manner. 
For now, we like to refer to [Wikipedia](https://en.wikipedia.org/wiki/Pattern_search_(optimization)) for more information such as pseudo code and visualizations in the search space.

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.problems.single import Himmelblau
from pymoo.optimize import minimize


problem = Himmelblau()

algorithm = PatternSearch()

res = minimize(problem,
               algorithm,
               verbose=False,
               seed=1)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.soo.nonconvex.pattern.PatternSearch
    :noindex:
```
