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

.. _nb_rastrigin:
```

## Rastrigin

The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima are regularly distributed. It is shown in the plot above in its two-dimensional form. 

+++

**Definition**

+++

\begin{align}
\begin{split}
f(x) &= 10n + \sum\limits_{i=1}^n {[x_i^2 - 10 \cos{(2 \pi x_i)}]} \\[2mm]
&-5.12 \leq x_i \leq 5.12 \quad i=1,\ldots,n
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

problem = get_problem("rastrigin", n_var=2)

FitnessLandscape(problem, angle=(45, 45), _type="surface").show()
```

```{code-cell} ipython3
FitnessLandscape(problem, _type="contour", colorbar=True).show()
```
