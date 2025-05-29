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

.. _nb_df:
```

## DF: Benchmark Problems for CEC2018 Competition on Dynamic Multiobjective Optimisation

+++

The problem suite is implemented based on <cite data-cite="df"></cite>.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_df1:
```

### DF1

```{code-cell} ipython3
import numpy as np

from pymoo.problems.dynamic.df import DF1
from pymoo.visualization.scatter import Scatter

plot = Scatter()

for t in np.linspace(0, 10.0, 100):
    problem = DF1(time=t)
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)

plot.show()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_df2:
```

### DF2

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF2

plot = Scatter()

for t in np.linspace(0, 10.0, 100):
    problem = DF2(time=t)
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)

plot.show()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_df3:
```

### DF3

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF3

plot = Scatter()

for t in np.linspace(0, 10.0, 100):
    problem = DF3(time=t)
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)

plot.show()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_df4:
```

### DF4

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF4

plot = Scatter()

for t in np.linspace(0, 10.0, 100):
    problem = DF4(time=t)
    plot.add(problem.pareto_front() + 2*t, plot_type="line", color="black", alpha=0.7)

plot.show()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_df5:
```

### DF5

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF5

plot = Scatter()

for t in np.linspace(0, 2.0, 100):
    problem = DF5(time=t)
    plot.add(problem.pareto_front(n_pareto_points=300) + 2*t, plot_type="line", color="black", alpha=0.7)

plot.show()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_df6:
```

### DF6

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF6

plot = Scatter()

for t in np.linspace(0, 2.0, 100):
    problem = DF6(time=t)
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)

plot.show()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_df7:
```

### DF7

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF7

plot = Scatter()

for t in np.linspace(0, 1.0, 20):
    problem = DF7(time=t)
    plot.add(problem.pareto_front() + 2*t, plot_type="line", color="black", alpha=0.7)

plot.show()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_df8:
```

### DF8

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF8

plot = Scatter()

for t in np.linspace(0, 2.0, 20):
    problem = DF8(time=t)
    plot.add(problem.pareto_front() + 4*t, plot_type="line", color="black", alpha=0.7)

plot.show()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_df9:
```

### DF9

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF9

plot = Scatter()

for t in np.linspace(0, 2.0, 20):
    problem = DF9(time=t)
    plot.add(problem.pareto_front() + 2*t, plot_type="line", color="black", alpha=0.7)

plot.show()
```

### DF10

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF10
import matplotlib.pyplot as plt

for t in [0.0, 1.0, 1.5, 2.0]:
    
    plot = Scatter()
    problem = DF10(time=t)
    plot.add(problem.pareto_front() + 2*t, plot_type="line", color="black", alpha=0.7)
    plot.do()
    plt.show()
    
print("DONE")
```

### DF11

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF11
import matplotlib.pyplot as plt

for t in [0.0, 1.0, 1.5, 2.0]:
    
    plot = Scatter()
    problem = DF11(time=t)
    plot.add(problem.pareto_front() + 2*t, plot_type="line", color="black", alpha=0.7)
    plot.do()
    plt.show()
    
print("DONE")
```

### DF12

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF12
import matplotlib.pyplot as plt

for t in [0.0, 0.1, 0.2]:
    
    plot = Scatter()
    problem = DF12(time=t)
    plot.add(problem.pareto_front() + 2*t, color="black", alpha=0.7)
    plot.do()
    plt.show()
    
print("DONE")
```

### DF13

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF13
import matplotlib.pyplot as plt

for t in [0.0, 0.2, 0.3, 0.4]:
    
    plot = Scatter()
    problem = DF13(time=t)
    plot.add(problem.pareto_front() + 2*t, color="black", alpha=0.7)
    plot.do()
    plt.show()
    
print("DONE")
```

### DF14

```{code-cell} ipython3
from pymoo.problems.dynamic.df import DF14
import matplotlib.pyplot as plt

for t in [0.0, 0.2, 0.5, 1.0]:
    
    plot = Scatter()
    problem = DF14(time=t)
    plot.add(problem.pareto_front() + 2*t, color="black", alpha=0.7)
    plot.do()
    plt.show()
    
print("DONE")
```
