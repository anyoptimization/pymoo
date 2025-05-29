---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

# SYM-PART

The SYM-PART <cite data-cite="sym_part"></cite> problem suite is a multi-modal multi-objective optimization problem (MMOP).
**In MMOPs, a solution $y$ in the objective space may have several inverse images in the decision space**.
For this reason, an MMOP could have more than one Pareto subsets.

The SYM-PART has two variants: SYM-PART simple and SYM-PART rotated. Both of them have the same Pareto front.
But their Pareto sets are different.

+++

## 1. SYM-PART Simple

+++

### Pareto subsets

```{code-cell} ipython3
from pymoo.problems.multi.sympart import SYMPART, SYMPARTRotated
from pymoo.visualization.scatter import Scatter

problem = SYMPART()
ps = problem.pareto_set()
Scatter(title="Pareto set", xlabel="$x_1$", ylabel="$x_2$").add(ps).show()
```

### Pareto front

```{code-cell} ipython3
pf = problem.pareto_front()
Scatter(title="Pareto front").add(pf).show()
```

## 2. SYM-PART Rotated
### Pareto subsets
The Pareto subsets can be rotated.

```{code-cell} ipython3
from numpy import pi

# rotate pi/3 counter-clockwisely
problem = SYMPARTRotated(angle=pi/3)
ps = problem.pareto_set()
Scatter(title="Pareto set", xlabel="$x_1$", ylabel="$x_2$").add(ps).show()
```

### Pareto front

```{code-cell} ipython3
pf = problem.pareto_front()
Scatter(title="Pareto front").add(pf).show()
```
