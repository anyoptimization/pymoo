---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: default
  language: python
  name: default
---

+++ {"raw_mimetype": "text/restructuredtext"}

.. _nb_kktpm:

+++

## Karush Kuhn Tucker Proximity Measure (KKTPM)




In 2016, Deb and Abouhawwash proposed Karush Kuhn Tucker Proximity Measure (KKTPM) <cite data-cite="kktpm1"></cite>, a metric that can measure how close a point is from being “an optimum”. The smaller the metric, the closer the point. This does not require the Pareto front to be known, but the gradient information needs to be approximated.
Their metric applies to both single objective and multi-objective optimization problems. 

In a single objective problem, the metric shows how close a point is from being a “local optimum”, while in multi-objective problems, the metric shows how close a point is from being a “local Pareto point”. Exact calculations of KKTPM for each point requires solving a whole optimization problem, which is extremely time-consuming. To avoid this problem, the authors of the original work again proposed several approximations to the true KKTPM, namely Direct KKTPM, Projected KKTPM, Adjusted KKTPM, and Approximate KKTPM. Approximate KKTPM is simply the average of the former three and is what we call simply “KKTPM”. Moreover, they were able to show that Approximate KKTPM is reliable and can be used in place of the exact one <cite data-cite="kktpm2"></cite>.

+++

<div style="text-align: center;">
    <img src="https://github.com/anyoptimization/pymoo-data/blob/main/docs/images/kktpm.png?raw=true" width="350">
</div>

+++

Let us now see how to use pymoo to calculate the KKTPM for a point:

```{code-cell} ipython3
from pymoo.constraints.from_bounds import ConstraintsFromBounds
from pymoo.gradient.automatic import AutomaticDifferentiation
from pymoo.problems import get_problem

problem = AutomaticDifferentiation(ConstraintsFromBounds(get_problem("zdt1", n_var=30)))
```

For instance, the code below calculates the KKTPM metric for randomly sampled points for the given example:

```{code-cell} ipython3
from pymoo.indicators.kktpm import KKTPM
from pymoo.operators.sampling.rnd import FloatRandomSampling

X = FloatRandomSampling().do(problem, 100).get("X")
kktpm = KKTPM().calc(X, problem)
```

Moreover, a whole run of a genetic algorithm can be analyzed by storing each generation's history and then calculating the KKTPM metric for each of the points:

```{code-cell} ipython3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.evaluator import Evaluator


algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)

# make sure each evaluation also has the derivatives - necessary for KKTPM
evaluator = Evaluator(evaluate_values_of=["F", "G", "dF", "dG"])

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               evaluator=evaluator,
               seed=1,
               save_history=True,
               verbose=False)
```

```{code-cell} ipython3
import pandas as pd
import numpy as np

# Collect KKTPM data for each generation
data = []
for algorithm in res.history:
    if algorithm.n_gen % 5 == 0:
        X = algorithm.pop.get("X")
        kktpm = KKTPM().calc(X, problem)
        
        # Add each individual's KKTPM value with generation info
        for i, value in enumerate(kktpm):
            data.append({
                'generation': algorithm.n_gen,
                'individual': i,
                'kktpm': value
            })

# Create DataFrame
df = pd.DataFrame(data)
df
```

```{code-cell} ipython3
# Get summary statistics for each generation
stats_by_gen = df.groupby('generation')['kktpm'].describe()
stats_by_gen
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Plot the quartiles and median over generations
generations = stats_by_gen.index
plt.plot(generations, stats_by_gen['25%'], label="Q1 (25th percentile)")
plt.plot(generations, stats_by_gen['50%'], label="Median")
plt.plot(generations, stats_by_gen['75%'], label="Q3 (75th percentile)")
plt.yscale("log")
plt.xlabel("Generation")
plt.ylabel("KKTPM")
plt.title("KKTPM Statistics Over Generations")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
