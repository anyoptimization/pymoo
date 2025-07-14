---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_cmopso:
```

.. meta::
   :description: An implementation of the Competitive Mechanism based Multi-objective Particle Swarm Optimizer (CMOPSO) algorithm. The algorithm uses a competitive learning strategy with binary tournament selection on elites to maintain diversity and convergence.

+++

.. meta::
   :keywords: CMOPSO, Particle Swarm Optimization, Multi-objective Optimization, Competitive Mechanism, Python

+++

# CMOPSO: Competitive Mechanism based Multi-objective Particle Swarm Optimizer

+++

The algorithm is implemented based on <cite data-cite="cmopso"></cite>.
CMOPSO extends the traditional Particle Swarm Optimization (PSO) framework to handle multi-objective optimization problems through a competitive learning mechanism. The key innovation lies in the competitive mechanism that selects leaders for particles through binary tournaments on elite solutions, promoting both convergence and diversity.

+++

## Key Features

**Competitive Learning Strategy**: Each particle learns from a "winner" selected from the elite archive through binary tournament selection. The winner is chosen based on the smallest angle between the particle's current position and the elite's position, promoting diversity.

**Elite Archive Management**: Uses an external archive to store non-dominated solutions with a crowding distance-based tournament survival strategy to maintain diversity.

**Velocity and Position Updates**: Standard PSO velocity update with competitive leader selection, followed by polynomial mutation for enhanced exploration.

+++

### Algorithm Overview

1. **Initialization**: Initialize population and velocities randomly
2. **Competitive Leader Selection**: For each particle, select a leader from elite archive using binary tournament based on angular distance
3. **Velocity Update**: Update velocity using competitive learning from selected leader
4. **Position Update**: Update particle positions
5. **Mutation**: Apply polynomial mutation for diversity
6. **Archive Update**: Update elite archive with new solutions
7. **Survival Selection**: Use SPEA2 survival strategy to select next generation

+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = CMOPSO(
    pop_size=100,
    max_velocity_rate=0.2,
    elite_size=10,
    mutation_rate=0.5,
    seed=1
)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
```

+++

### Customization

CMOPSO can be customized with different parameters to suit specific problem characteristics:

```{code-cell} ipython3
from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt2")

# Customized CMOPSO with different parameters
algorithm = CMOPSO(
    pop_size=150,           # Larger population for better diversity
    max_velocity_rate=0.3,  # Higher velocity for more exploration
    elite_size=20,          # Larger elite archive
    mutation_rate=0.3,      # Lower mutation rate for more exploitation
    initial_velocity="zero", # Start with zero velocity
    seed=1
)

res = minimize(problem,
               algorithm,
               ('n_gen', 300),
               seed=1,
               verbose=False)

Scatter().add(res.F, facecolor="none", edgecolor="blue").show()
```

+++

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.cmopso.CMOPSO
    :noindex:
```

+++

### References

<cite data-cite="cmopso">Zhang, X., Zheng, X., Cheng, R., Qiu, J., & Jin, Y. (2018). A competitive mechanism based multi-objective particle swarm optimizer with fast convergence. Information Sciences, 427, 63-76.</cite>

+++

### Implementation

This algorithm has been implemented by [Gideon Oludeyi](https://github.com/gideonoludeyi) and is based on the original paper by Zhang et al. (2018). The implementation follows the competitive learning strategy with binary tournament selection on elites to maintain diversity and convergence in multi-objective optimization problems. 