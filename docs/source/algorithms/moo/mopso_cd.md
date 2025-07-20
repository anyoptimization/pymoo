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

.. _nb_mopso_cd:
```

.. meta::
   :description: An implementation of Multi-Objective Particle Swarm Optimization with Crowding Distance (MOPSO-CD) algorithm. The algorithm extends MOPSO with crowding distance mechanism for leader selection and archive management to ensure well-distributed Pareto fronts.

+++

.. meta::
   :keywords: MOPSO-CD, Particle Swarm Optimization, Multi-objective Optimization, Crowding Distance, Archive Management, Python

+++

# MOPSO-CD: Multi-Objective Particle Swarm Optimization with Crowding Distance

+++

MOPSO-CD extends the traditional Particle Swarm Optimization (PSO) framework to handle multi-objective optimization problems through enhanced diversity mechanisms. The algorithm incorporates crowding distance-based leader selection and archive management to ensure a well-distributed Pareto front, making it particularly suitable for problems requiring good diversity preservation.

+++

## Key Features

**Crowding Distance-based Leader Selection**: Uses crowding distance to select diverse leaders for particles, ensuring better exploration of the objective space.

**External Archive Management**: Maintains an external archive of non-dominated solutions with crowding distance-based pruning to preserve diversity.

**Tournament Selection**: Implements binary tournament selection with crowding distance to maintain diversity while keeping solution quality.

**Enhanced Exploration**: Increased inertia weight and velocity bounds for better exploration capabilities.

+++

### Algorithm Overview

1. **Initialization**: Initialize population, velocities, and personal bests
2. **Archive Management**: Update external archive with non-dominated solutions using crowding distance pruning
3. **Diverse Leader Selection**: Select leaders for particles using crowding distance-based tournament selection
4. **Velocity Update**: Update velocities using cognitive and social components with selected leaders
5. **Position Update**: Update particle positions with velocity bounds
6. **Personal Best Update**: Update personal best solutions based on dominance
7. **Archive Pruning**: Maintain archive size using crowding distance-based selection

+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = MOPSO_CD(
    pop_size=100,
    w=0.6,
    c1=2.0,
    c2=2.0,
    max_velocity_rate=0.5,
    archive_size=200,
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

MOPSO-CD can be customized with different parameters to balance exploration and exploitation:

```{code-cell} ipython3
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt2")

# Customized MOPSO-CD with different parameters
algorithm = MOPSO_CD(
    pop_size=150,           # Larger population for better diversity
    w=0.729844,            # Standard PSO inertia weight
    c1=1.49618,            # Cognitive parameter
    c2=1.49618,            # Social parameter
    max_velocity_rate=0.2, # Lower velocity for more exploitation
    archive_size=100,      # Smaller archive for faster convergence
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

### Application to Multi-Objective Reinforcement Learning

MOPSO-CD is particularly well-suited for multi-objective reinforcement learning problems like MO-HalfCheetah, where maintaining diversity in the objective space is crucial for finding a well-distributed set of trade-off solutions.

+++

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.mopso_cd.MOPSO_CD
    :noindex:
```

+++

### Algorithm Comparison

MOPSO-CD differs from other multi-objective PSO variants in several key aspects:

- **Leader Selection**: Uses crowding distance-based tournament selection instead of random selection
- **Archive Management**: Implements sophisticated archive pruning using crowding distance
- **Exploration Focus**: Higher inertia weight and velocity bounds for better exploration
- **Diversity Preservation**: Enhanced mechanisms for maintaining solution diversity

+++

### References

The algorithm is based on extensions of traditional MOPSO with crowding distance mechanisms for improved diversity preservation in multi-objective optimization problems.

+++

### Implementation

This algorithm has been implemented by [Rasa Khosrowshahi](https://github.com/rkhosrowshahi) and extends traditional MOPSO with crowding distance mechanisms for leader selection and archive management. The implementation is particularly well-suited for multi-objective reinforcement learning problems where maintaining diversity in the objective space is crucial for finding well-distributed trade-off solutions. 
