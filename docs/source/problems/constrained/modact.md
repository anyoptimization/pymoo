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
.. _nb_modact:
```

+++ {"pycharm": {"name": "#%% md\n"}}

# MODAct

MODAct (multi-objective design of actuators) is a real-world benchmark test-suite <cite data-cite="modact"></cite> for constrained multi-objective optimization. The optimization problems aim at finding small electro-actuators given some objectives and constraints. Currently, there are 20 problems with up to 5 objectives and 10 inequality constraints, summarized in the table below.

In order to solve these problems, you will need to have the [modact](https://github.com/epfl-lamd/modact) package and its dependencies installed (Docker image available). A single solution evaluation takes about 20 ms. Therefore, the use of parallel schemes is advised. 


The estimated Pareto-fronts for CS1-4 and CT-4 have been added to pymoo directly. The others because of their file sizes have to be downloaded [here](https://doi.org/10.5281/zenodo.3824302), and used during initialization as shown below.

For more information please refer to the associated publication <cite data-cite="modact"></cite>.

|Problem|Variables|Objectives|Constraints|
|:---|:---|:---|:---|
| CS1   | 20      |  2       | 7         |
| CS2   | 20      |  2       | 8         |  
| CS3   | 20      |  2       | 10        | 
| CS4   | 20      |  2       | 9         |
| CT1, CTS1, CTSE1, CTSEI1  | 20  |  2,3,4 or 5  | 7  |  
| CT2, CTS2, CTSE2, CTSEI2  | 20  |  2,3,4 or 5  | 8  |  
| CT3, CTS3, CTSE3, CTSEI3  | 20  |  2,3,4 or 5  | 10 |  
| CT4, CTS4, CTSE4, CTSEI4  | 20  |  2,3,4 or 5  | 9  |  

Some usage examples are highlighted in the following sections.

```{raw-cell}
---
pycharm:
  name: '#%% raw

    '
raw_mimetype: text/restructuredtext
---
.. _nb_modact_cs3:
```

+++ {"pycharm": {"name": "#%% md\n"}}

### CS3

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
from pymoo.problems.multi import MODAct
from pymoo.visualization.util import plot

problem = MODAct("cs3")
plot(problem.pareto_front(), no_fill=True)
```

+++ {"pycharm": {"name": "#%% md\n"}}

### CT1

```{code-cell}
---
pycharm:
  name: '#%%

    '
---
from pymoo.visualization.util import plot

problem = MODAct("ct1")
plot(problem.pareto_front(), no_fill=True)
```

+++ {"pycharm": {"name": "#%% md\n"}}

<sub>Implementation by [the author (cyrilpic)](https://github.com/cyrilpic).
