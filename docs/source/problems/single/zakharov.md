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

.. _nb_zakharov:
```

## Zakharov

The Zakharov function has no local minima except the global one. It is shown here in its two-dimensional form.

+++

**Definition**

+++

\begin{align}
\begin{split}
f(x) &= \sum\limits_{i=1}^n {x_i^2} + \bigg( \frac{1}{2} \sum\limits_{i=1}^n {ix_i} \bigg)^2 + \bigg( \frac{1}{2} \sum\limits_{i=1}^n {ix_i} \bigg)^4, \\[2mm]
&-10 \leq x_i \leq 10 \quad i=1,\ldots,n
\end{split}
\end{align}

+++

**Optimum**

+++

$$f(x^*) = 0 \; \text{at} \; x^* = (0,\ldots,0) $$

+++

**Fitness Landscape**

```{code-cell} ipython3
import numpy as np

from pymoo.problems import get_problem
from pymoo.visualization.fitness_landscape import FitnessLandscape

problem = get_problem("zakharov", n_var=2)

FitnessLandscape(problem, angle=(45, 45), _type="surface").show()
```

```{code-cell} ipython3
FitnessLandscape(problem, _type="contour", contour_levels=200, colorbar=True).show()
```
