import matplotlib.pyplot as plt

from pymoo.algorithms.moo.omni import OmniOptimizer
from pymoo.core.callback import Callback
from pymoo.operators.crossover.pcx import PCX
from pymoo.optimize import minimize
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.visualization.scatter import Scatter

# The Omni-Optimizer is a standard genetic algorithm, so several of the extensions
# suggested as "future work" in the original presentation are simply different
# *compositions* of existing pymoo components - the algorithm itself is not changed.

problem = OmniTest(n_var=2)

# ---------------------------------------------------------------------------------------
# Using PCX instead of SBX - just pass a different crossover operator
# ---------------------------------------------------------------------------------------
res_pcx = minimize(problem,
                   OmniOptimizer(pop_size=100, crossover=PCX()),
                   ('n_gen', 200), seed=1, verbose=False)


# ---------------------------------------------------------------------------------------
# Self-adapting parameters over the run via a Callback - also without touching the
# algorithm. This increases the polynomial mutation index (eta_m) for finer mutations
# ("arbitrary precision") and decreases the loose-dominance margin (delta) to
# progressively tighten the fronts. The callback assumes the default operators of
# OmniOptimizer (PM mutation and the loose-dominance survival).
# ---------------------------------------------------------------------------------------
class ParameterControl(Callback):

    def __init__(self, n_max_gen, eta=(20.0, 100.0), delta=(1e-2, 1e-4)):
        super().__init__()
        self.n_max_gen = n_max_gen
        self.eta = eta
        self.delta = delta

    def notify(self, algorithm):
        # progress in [0, 1] over the run
        t = min(1.0, (algorithm.n_gen or 1) / self.n_max_gen)

        # increase the polynomial mutation index eta_m (linear)
        algorithm.mating.mutation.eta.set(self.eta[0] + t * (self.eta[1] - self.eta[0]))

        # decrease the loose-dominance margin delta (geometric)
        algorithm.survival.nds.dominator.delta = self.delta[0] * (self.delta[1] / self.delta[0]) ** t


n_gen = 200
res_adapt = minimize(problem,
                     OmniOptimizer(pop_size=100),
                     ('n_gen', n_gen), seed=1,
                     callback=ParameterControl(n_gen), verbose=False)

# both compositions still recover all equivalent Pareto subsets
PS = problem.pareto_set(5000)

plot = Scatter(title="PCX crossover", labels="x")
plot.add(PS, s=10, color="red", label="Pareto set")
plot.add(res_pcx.X, s=20, color="blue", label="Obtained solutions")
plot.do()
plt.legend()

plot = Scatter(title="Self-adapting eta_m and delta", labels="x")
plot.add(PS, s=10, color="red", label="Pareto set")
plot.add(res_adapt.X, s=20, color="blue", label="Obtained solutions")
plot.do()
plt.legend()

plt.show()
