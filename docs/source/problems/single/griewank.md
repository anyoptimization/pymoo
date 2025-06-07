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

.. _nb_griewank:
```

## Griewank

The Griewank function has many widespread local minima, which are regularly distributed. The complexity is shown in the zoomed-in plots. 

+++

**Definition**

+++

\begin{align}
\begin{split}
f(x) & = & \; \sum_{i=1}^n \; \frac{x_i^2}{4000} - \prod_{i=1}^n \cos \Bigg( \frac{x_i}{\sqrt{i}} \Bigg) + 1 \\[4mm]
&&-600 \leq x_i \leq 600 \quad i=1, \ldots, n \\[4mm]
\end{split}
\end{align}

+++

**Optimum**

+++

$$f(x^*) = 0 \; \text{at} \; x^* = (0,\ldots,0) $$

+++

**Contour**

```{code-cell} ipython3
from pymoo.problems import get_problem
from pymoo.visualization.fitness_landscape import FitnessLandscape

problem = get_problem("griewank", n_var=1)
plot = FitnessLandscape(problem, _type="surface", n_samples=1000)
plot.do()
plot.apply(lambda ax: ax.set_xlim(-200, 200))
plot.apply(lambda ax: ax.set_ylim(-1, 13))
plot.show()
```
