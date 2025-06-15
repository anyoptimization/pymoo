---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

+++ {"raw_mimetype": "text/restructuredtext"}

.. _nb_repair:

+++

# Repair

The repair operator is mostly problem-dependent. Most commonly, it is used to make sure the algorithm is only searching in the feasible space. It is applied after the offspring have been reproduced. In the following, we are using the knapsack problem to demonstrate the repair operator in *pymoo*.

+++

In the well-known **Knapsack Problem**. In this problem, a knapsack has to be filled with items without violating the maximum weight constraint. Each item $j$ has a value $b_j \geq 0$  and a weight $w_j \geq 0$ where $j \in \{1, .., m\}$. The binary decision vector $z = (z_1, .., z_m)$ defines, if an item is picked or not. The aim is to maximize the profit $g(z)$:

\begin{eqnarray}
max & & g(z) \\[2mm] \notag 
\text{s.t.} & & \sum_{j=1}^m z_j \, w_j \leq Q \\[1mm] \notag 
& & z = (z_1, .., z_m) \in \mathbb{B}^m \\[1mm] \notag 
g(z) & = & \sum_{j=1}^{m}  z_j \, b_j \\[2mm] \notag 
\end{eqnarray}


A simple GA will have some infeasible evaluations in the beginning and then concentrate on the feasible space.

```{code-cell} ipython3
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems.single.knapsack import create_random_knapsack_problem

problem = create_random_knapsack_problem(30)


algorithm = GA(pop_size=200,
               sampling=BinaryRandomSampling(),
               crossover=HUX(),
               mutation=BitflipMutation(),
               eliminate_duplicates=True)


res = minimize(problem,
               algorithm,
               termination=('n_gen', 10),
               verbose=True)
```


Because the constraint $\sum_{j=1}^m z_j \, w_j \leq Q$ is fairly easy to satisfy, we can make sure that this constraint is not violated by repairing the individual before evaluating the objective function.
A repair class has to be defined, and the population is given as input. The repaired population has to be returned.

```{code-cell} ipython3
import numpy as np
from pymoo.core.repair import Repair


class ConsiderMaximumWeightRepair(Repair):

    def _do(self, problem, Z, **kwargs):
        
        # maximum capacity for the problem
        Q = problem.C
        
        # the corresponding weight of each individual
        weights = (Z * problem.W).sum(axis=1)
        
        # now repair each indvidiual i
        for i in range(len(Z)):
            
            # the packing plan for i
            z = Z[i]
            
            # while the maximum capacity violation exists
            while weights[i] > Q:
                
                # randomly select an item currently picked
                item_to_remove = np.random.choice(np.where(z)[0])
                
                # and remove it
                z[item_to_remove] = False
                
                # adjust the weight
                weights[i] -= problem.W[item_to_remove]
          
        return Z
```

```{code-cell} ipython3
algorithm = GA(pop_size=200,
               sampling=BinaryRandomSampling(),
               crossover=HUX(),
               mutation=BitflipMutation(),
               repair=ConsiderMaximumWeightRepair(),
               eliminate_duplicates=True)


res = minimize(problem,
               algorithm,
               termination=('n_gen', 10),
               verbose=True)
```

As demonstrated, the repair operator makes sure no infeasible solution is evaluated. Even though this example seems to be quite easy, the repair operator makes especially sense for more complex constraints where domain-specific knowledge is known.
