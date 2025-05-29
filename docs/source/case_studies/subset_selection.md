---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

+++ {"raw_mimetype": "text/restructuredtext"}

.. _nb_subset_selection:

+++

## Subset Selection Problem

A genetic algorithm can be used to approach subset selection problems by defining custom operators. In general, a metaheuristic algorithm might not be the ultimate goal to implement in a real-world scenario; however, it might be useful to investigate patterns or characteristics of possible well-performing subsets. 
Let us consider a simple toy problem where we have to select numbers from a list. For every solution, exactly ten numbers have to be selected that their sum is minimized.
For the subset selection problem, a binary encoding can be used where **one** indicates a number is picked. In our problem formulation, the list of numbers is represented by $L$ and the binary encoded variable by $x$.

+++


\begin{align}
\begin{split}
\min f(x) & = & \sum_{k=1}^{n} L_k \cdot x_k\\[2mm]
\text{s.t.} \quad g(x)  & = & (\sum_{k=1}^{n} x_k - 10)^2\\[2mm]
\end{split}
\end{align}

+++

As shown above, the equality constraint is handled by ensuring $g(x)$ can only be zero if exactly ten numbers are chosen.
The problem can be implemented as follows:

```{code-cell} ipython3
import numpy as np
from pymoo.core.problem import ElementwiseProblem

class SubsetProblem(ElementwiseProblem):
    def __init__(self,
                 L,
                 n_max
                 ):
        super().__init__(n_var=len(L), n_obj=1, n_ieq_constr=1)
        self.L = L
        self.n_max = n_max

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(self.L[x])
        out["G"] = (self.n_max - np.sum(x)) ** 2
    
    
# create the actual problem to be solved
np.random.seed(1)
L = np.array([np.random.randint(100) for _ in range(100)])
n_max = 10
problem = SubsetProblem(L, n_max)
```

The customization requires writing custom operators in order to solve this problem efficiently. We recommend considering the feasibility directly in the evolutionary operators because, otherwise, most of the time, infeasible solutions will be processed.
The sampling creates a random solution where the subset constraint will always be satisfied. 
The mutation randomly removes a number and chooses another one. The crossover takes the values of both parents and then randomly picks either the one from the first or from the second parent until enough numbers are picked.

```{code-cell} ipython3
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False

        return X
```

After having defined the operators a genetic algorithm can be initialized.

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

algorithm = GA(
    pop_size=100,
    sampling=MySampling(),
    crossover=BinaryCrossover(),
    mutation=MyMutation(),
    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 60),
               seed=1,
               verbose=True)

print("Function value: %s" % res.F[0])
print("Subset:", np.where(res.X)[0])
```

Finally, we can compare the found subset with the optimum known simply through sorting:

```{code-cell} ipython3
opt = np.sort(np.argsort(L)[:n_max])
print("Optimal Subset:", opt)
print("Optimal Function Value: %s" % L[opt].sum())
```
