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

.. _nb_omni:
```

.. meta::
   :description: The Omni-Optimizer is a generic NSGA-II based evolutionary algorithm for single- and multi-objective, single- and multi-global optimization. It uses a dynamic epsilon-dominance, a crowding distance defined in both the objective and the variable space, and a restricted nearest-neighbor mating selection to find and maintain multiple equivalent Pareto-optimal solutions.

+++

.. meta::
   :keywords: Omni-Optimizer, Multi-modal Multi-objective Optimization, Epsilon Dominance, Crowding Distance, Variable Space, Multi-objective Optimization, Python

+++

# Omni-Optimizer

+++

The Omni-Optimizer <cite data-cite="omni_test"></cite> is a generic evolutionary algorithm that is able to solve single- and multi-objective problems with one or many (equivalent) optima. It is based on NSGA-II but introduces three modifications that make it particularly effective for **multi-modal multi-objective optimization**, i.e. problems where several distinct regions in the decision space map to the same (or an equivalent) region of the Pareto front.

+++

## Key Components

**Dynamic ε-dominance**: The non-dominated sorting uses a *loose* dominance relation. A solution only dominates another if it is better by more than a margin of `delta * epsilon_j` in at least one objective, where the per-objective `epsilon_j` is computed dynamically as the range of objective `j` in the current population. Solutions that are very close in objective space are therefore placed in the same front.

**Crowding distance in objective and variable space**: In addition to the usual objective-space crowding distance, a crowding distance is also computed in the decision (variable) space. For each solution, the larger of the two values is used whenever the solution is less crowded than the population average in *either* space, and the smaller one otherwise. This rewards solutions that are diverse in the decision space and is what allows the algorithm to keep equivalent solutions that would collapse under NSGA-II.

**Restricted nearest-neighbor mating**: Instead of pairing two random solutions, each binary tournament is held between a randomly selected solution and its nearest neighbor in the (normalized) decision space. This biases recombination towards solutions of the same basin and helps to preserve distinct optima.

Setting `var_crowding=False` essentially recovers NSGA-II, while `delta=0` disables the loose dominance and recovers the usual Pareto dominance.

+++

### Example

The `OmniTest` problem has `3^n_var` equivalent Pareto subsets in the decision space that all map to the same Pareto front. The Omni-Optimizer is able to find and maintain (almost) all of them.

```{code-cell} ipython3
from pymoo.algorithms.moo.omni import OmniOptimizer
from pymoo.optimize import minimize
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.visualization.scatter import Scatter

problem = OmniTest(n_var=2)

algorithm = OmniOptimizer(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter(title="Decision Space", labels="x")
plot.add(problem.pareto_set(5000), s=10, color="red", label="Pareto set")
plot.add(res.X, s=20, color="blue", label="Obtained solutions")
plot.show()
```

```{code-cell} ipython3
plot = Scatter(title="Objective Space")
plot.add(problem.pareto_front(5000), s=10, color="red", label="Pareto front")
plot.add(res.F, s=20, color="blue", label="Obtained solutions")
plot.show()
```

The effect of the variable-space niching becomes clear when it is disabled (`var_crowding=False`), which niches only in objective space and essentially recovers NSGA-II: the equivalent subsets collapse onto only a few of them.

```{code-cell} ipython3
res_no_var = minimize(problem,
                      OmniOptimizer(pop_size=100, var_crowding=False),
                      ('n_gen', 200),
                      seed=1,
                      verbose=False)

plot = Scatter(title="Decision Space (without variable-space niching)", labels="x")
plot.add(problem.pareto_set(5000), s=10, color="red", label="Pareto set")
plot.add(res_no_var.X, s=20, color="blue", label="Obtained solutions")
plot.show()
```

+++

### Customization

Because the Omni-Optimizer is a regular genetic algorithm, several extensions are simply different *compositions* of existing pymoo components and require no change to the algorithm. For example, the crossover operator can be swapped in a single line (e.g. `OmniOptimizer(crossover=PCX())`).

More elaborate ideas, such as adapting the parameters over the course of a run, can be implemented with a `Callback`. The following one increases the polynomial mutation index `eta_m` (for finer mutations towards the end) and decreases the loose-dominance margin `delta` (to progressively tighten the fronts), all without modifying the algorithm.

```{code-cell} ipython3
from pymoo.core.callback import Callback


class ParameterControl(Callback):

    def __init__(self, n_max_gen, eta=(20.0, 100.0), delta=(1e-2, 1e-4)):
        super().__init__()
        self.n_max_gen = n_max_gen
        self.eta = eta
        self.delta = delta

    def notify(self, algorithm):
        t = min(1.0, (algorithm.n_gen or 1) / self.n_max_gen)
        # increase the polynomial mutation index eta_m (linear)
        algorithm.mating.mutation.eta.set(self.eta[0] + t * (self.eta[1] - self.eta[0]))
        # decrease the loose-dominance margin delta (geometric)
        algorithm.survival.nds.dominator.delta = self.delta[0] * (self.delta[1] / self.delta[0]) ** t


n_gen = 200
res = minimize(problem,
               OmniOptimizer(pop_size=100),
               ('n_gen', n_gen),
               seed=1,
               callback=ParameterControl(n_gen),
               verbose=False)

plot = Scatter(title="Self-adapting eta_m and delta", labels="x")
plot.add(problem.pareto_set(5000), s=10, color="red", label="Pareto set")
plot.add(res.X, s=20, color="blue", label="Obtained solutions")
plot.show()
```

+++

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.omni.OmniOptimizer
    :noindex:
```

+++

### Acknowledgement

The Omni-Optimizer implementation in pymoo was contributed by [evanroyrees](https://github.com/evanroyrees), faithfully following the reference implementation of the original authors (Deb and Tiwari, 2008).

*Thank you for the contribution — pymoo grows through community contributions like this one.*
