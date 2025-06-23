import numpy as np

from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.decomposition.asf import ASF
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT
from pymoo.problems.util import decompose
from pymoo.visualization.scatter import Scatter


class ModifiedZDT1(ZDT):

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        out_of_bounds = np.any(set_to_bounds_if_outside_by_problem(self, x.copy()) != x)

        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum((x[:, 1:]) ** 2, axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        if out_of_bounds:
            f1 = np.full(x.shape[0], np.inf)
            f2 = np.full(x.shape[0], np.inf)

        out["F"] = np.column_stack([f1, f2])


n_var = 2
original_problem = ModifiedZDT1(n_var=n_var)
weights = np.array([0.5, 0.5])

decomp = ASF(ideal_point=np.array([0.0, 0.0]), nadir_point=np.array([1.0, 1.0]))

pf = original_problem.pareto_front()

problem = decompose(original_problem,
                    decomp,
                    weights
                    )


for i in range(100):

    if i != 23:
        continue

    algorithm = NelderMead(n_max_restarts=10, adaptive=True)

    res = minimize(problem,
                   algorithm,
                   #scipy_minimize("Nelder-Mead"),
                   #termination=("n_eval", 30000),
                   seed=i,
                   verbose=False)

    #print(res.X)

    F = ModifiedZDT1(n_var=n_var).evaluate(res.X, return_values_of="F")
    print(i, F)

opt = decomp.do(pf, weights).argmin()



print(pf[opt])
print(decomp.do(pf, weights).min())

plot = Scatter()
plot.add(pf)
plot.add(F)
plot.add(np.vstack([np.zeros(2), weights]), plot_type="line")
plot.add(pf[opt])
plot.show()
