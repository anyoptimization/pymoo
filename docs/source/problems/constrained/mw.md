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

.. _nb_mw:
```

## MW

MW <cite data-cite="mw"></cite> is a constrained multi-objective test suite constructed in a similar fashion to CTP or WFG with 3 different distance functions and 3 local adjustment methods. Most problems are biobjective problems, except MW4, MW8 and MW14 which are scalable ($m \geq 3$).

They aim at replacing the CTP test suite by proposing more complex problems with up to 4 inequality constraints.

```{code-cell} ipython3
from pymoo.problems import get_problem
from pymoo.visualization.util import plot
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_mw1:
```

### MW1

```{code-cell} ipython3


problem = get_problem("mw1")
plot(problem.pareto_front(), no_fill=True)
```

### MW2

```{code-cell} ipython3
problem = get_problem("mw2")
plot(problem.pareto_front(), no_fill=True)
```

### MW3

```{code-cell} ipython3
problem = get_problem("mw3")
plot(problem.pareto_front(), no_fill=True)
```

### MW4

```{code-cell} ipython3
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
pf = get_problem("mw4").pareto_front(ref_dirs)
Scatter(angle=(45,45)).add(pf, color="red").show()
```

### MW5

```{code-cell} ipython3
problem = get_problem("mw5")
plot(problem.pareto_front(), no_fill=True)
```

### MW6

```{code-cell} ipython3
problem = get_problem("mw6")
plot(problem.pareto_front(), no_fill=True)
```

### MW7

```{code-cell} ipython3
problem = get_problem("mw7")
plot(problem.pareto_front(), no_fill=True)
```

### MW8

```{code-cell} ipython3
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
pf = get_problem("mw8").pareto_front(ref_dirs)
Scatter(angle=(45,45)).add(pf, color="red").show()
```

### MW9

```{code-cell} ipython3
problem = get_problem("mw9")
plot(problem.pareto_front(), no_fill=True)
```

### MW10

```{code-cell} ipython3
problem = get_problem("mw10")
plot(problem.pareto_front(), no_fill=True)
```

### MW11

```{code-cell} ipython3
problem = get_problem("mw11")
plot(problem.pareto_front(), no_fill=True)
```

### MW12

```{code-cell} ipython3
problem = get_problem("mw12")
plot(problem.pareto_front(), no_fill=True)
```

### MW13

```{code-cell} ipython3
problem = get_problem("mw13")
plot(problem.pareto_front(), no_fill=True)
```

### MW14

```{code-cell} ipython3
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
pf = get_problem("mw14").pareto_front()
Scatter(angle=(45,45)).add(pf, color="red").show()
```

<sub>Python implementation by [cyrilpic](https://github.com/cyrilpic) based on the [original C++ code](http://www.escience.cn/people/yongwang1/index.html).</sub>
