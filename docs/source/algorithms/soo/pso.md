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

.. _nb_pso:
```

.. meta::
   :description: An implementation of the famous Particle Swarm Optimization (PSO) algorithm which is inspired by the behavior of the movement of particles represented by their position and velocity. Each particle is updated considering the cognitive and social behavior in a swarm.

+++

.. meta::
   :keywords: Particle Swarm Optimization, Nature-inspired Algorithm, Single-objective Optimization, Python

+++

# PSO: Particle Swarm Optimization

+++

Particle Swarm Optimization was proposed in 1995 by Kennedy and Eberhart <cite data-cite="pso"></cite> based on the simulation of social behavior. The algorithm uses a *swarm* of particles to guide its search. Each particle has a velocity and is influenced by locally and globally best-found solutions. Many different implementations have been proposed in the past and, therefore, it is quite difficult to refer to THE correct implementation of PSO. However, the general concepts shall be explained in the following.

Given the following variables:

- $X_{d}^{(i)}$ d-th coordinate of i-th particle's position
- $V_{d}^{(i)}$ d-th coordinate of i-th particle's velocity 
- $\omega$ Inertia weight
- $P_{d}^{(i)}$ d-th coordinate of i-th particle's *personal* best 
- $G_{d}^{(i)}$ d-th coordinate of the globally (sometimes also only locally) best solution found
- $c_1$ and $c_2$ Two weight values to balance exploiting the particle's best $P_{d}^{(i)}$ and swarm's best $G_{d}^{(i)}$ 
- $r_1$ and $r_2$ Two random values being created for the velocity update

The velocity update is given by:

+++

\begin{equation}
V_{d}^{(i)} = \omega \, V_{d}^{(i)} \;+\; c_1 \, r_1 \, \left(P_{d}^{(i)} - X_{d}^{(i)}\right) \;+\; c_2 \, r_2 \, \left(G_{d}^{(i)} - X_{d}^{(i)}\right)
\end{equation}

+++

The corresponding position value is then updated by:

+++

\begin{equation}
X_{d}^{(i)} = X_{d}^{(i)} + V_{d}^{(i)}
\end{equation}

+++

The social behavior is incorporated by using the *globally* (or locally) best-found solution in the swarm for the velocity update. Besides the social behavior, the swarm's cognitive behavior is determined by the particle's *personal* best solution found.
The cognitive and social components need to be well balanced to ensure that the algorithm performs well on a variety of optimization problems.
Thus, some effort has been made to determine suitable values for $c_1$ and $c_2$. In **pymoo** both values are updated as proposed in <cite data-cite="pso_adapative"></cite>. Our implementation deviates in some implementation details (e.g. fuzzy state change) but follows the general principles proposed in the paper. 

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.problems.single import Rastrigin
from pymoo.optimize import minimize

problem = Rastrigin()

algorithm = PSO()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.soo.nonconvex.pso.PSO
    :noindex:
    :no-undoc-members:
```
