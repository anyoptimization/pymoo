---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

# Omni-test
The Omni-test problem is a multi-modal multi-objective optimization problem proposed by Deb in <cite data-cite="omni_test"></cite>. It has two objective
functions. Suppose that the dimension of the decision space is $D$, then it has $3^D$ Pareto subsets in the decision
space corresponding to the same Pareto front.

+++

## 2-dimensional case
### Pareto front

```{code-cell} ipython3
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.visualization.scatter import Scatter

problem = OmniTest(n_var=2)
pf = problem.pareto_front(1000)
Scatter(title="Pareto front").add(pf).show()
```

### Pareto set

```{code-cell} ipython3
ps = problem.pareto_set(1000)
Scatter(title="Pareto set", labels=["$x_1$", "$x_2$"]).add(ps).show()
```

## 3-dimensional case
### Pareto front

```{code-cell} ipython3
problem = OmniTest(n_var=3)
pf = problem.pareto_front(3000)
Scatter(title="Pareto front").add(pf).show()
```

### Pareto set

```{code-cell} ipython3
import matplotlib.pyplot as plt

ps = problem.pareto_set(1000)
sc = Scatter(title="Pareto set", labels=["$x_1$", "$x_2$", "$x_3$"])
sc.add(ps)
sc.do()
sc.ax.view_init(elev=20, azim=5)
plt.show()
```
