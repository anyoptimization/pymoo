---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

```{raw-cell}
---
pycharm:
  name: '#%% raw

    '
raw_mimetype: text/restructuredtext
---
.. _nb_dascmop:
```

+++ {"pycharm": {"name": "#%% md\n"}}

## DAS-CMOP

DAS-CMOP is a constrained multi-objective test suite with tunable constraints <cite data-cite="dascmop"></cite>. The constraints are tuned using a difficulty triplet: $(\eta, \zeta, \gamma)$, with $\eta, \zeta, \gamma \in [0, 1]$. The triplet allows to adjust the diversity, the feasibility and the convergence hardness respectively.

There are 6 bi-objective problems DAS-CMOP1-6 (11 constraints) and 3 three-objective problems DAS-CMOP7-9 (7 constraints). Each of these can be initialized with a custom difficulty triplet or the authors proposed a set of 16 triplets:

|No. | Difficulty     | No. | Difficulty     | No | Difficulty     | No | Difficulty
|---|---|---|---|---|---|---|---
| 1  | (0.25,0.0,0.0) | 2   | (0.0,0.25,0.0) | 3  | (0.0,0.0,0.25) | 4  | (0.25,0.25,0.25)
| 5  | (0.5,0.0,0.0)  | 6   | (0.0,0.5,0.0)  | 7  | (0.0,0.0,0.5)  | 8  | (0.5,0.5,0.5)
| 9  | (0.75,0.0,0.0) | 10  | (0.0,0.75,0.0) | 11 | (0.0,0.0,0.75) | 12 | (0.75,0.75,0.75)
| 13 | (0.0,1.0,0.0)  | 14  | (0.5,1.0,0.0)  | 15 | (0.0,1.0,0.5)  | 16 | (0.5,1.0,0.5)

The Pareto fronts are different for each triplet.

```{raw-cell}
---
pycharm:
  name: '#%% raw

    '
raw_mimetype: text/restructuredtext
---
.. _nb_dascmop1:
```

+++ {"pycharm": {"name": "#%% md\n"}}

### DAS-CMOP1 (1)

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
from pymoo.problems import get_problem
from pymoo.visualization.util import plot

problem = get_problem("dascmop1", 1)
plot(problem.pareto_front(), no_fill=True)
```

+++ {"pycharm": {"name": "#%% md\n"}}



+++ {"pycharm": {"name": "#%% md\n"}}

### DAS-CMOP7 (12)

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

pf = get_problem("dascmop7", 12).pareto_front()
Scatter(angle=(45, 45)).add(pf, color="red").show()
```

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

pf = get_problem("dascmop7", 12).pareto_front()
Scatter(angle=(45, 45)).add(pf, color="red").show()
```

+++ {"pycharm": {"name": "#%% md\n"}}

<sub>Python implementation by [cyrilpic](https://github.com/cyrilpic) based on the [original JAVA code](http://imagelab.stu.edu.cn/Content.aspx?type=content&Content_ID=1310).</sub>
