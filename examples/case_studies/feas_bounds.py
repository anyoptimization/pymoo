import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize

PLOT = False
SEED = 1

n_points = 1000
n_dim = 2

np.random.seed(SEED)

xl = np.array([-1, -10])
xu = np.array([1, 10])

X = xl + (np.random.uniform(size=(n_points, n_dim)) * (xu - xl))
a, b = X.T
p = np.minimum(1.0, a ** 2 + (1 / 10 * b) ** 2)
p[p < 0.1] = 0.0

y = ~(np.random.uniform(size=n_points) < p)

if PLOT:
    plt.scatter(a[y], b[y], facecolor="none", edgecolors="green", label="+")
    plt.scatter(a[~y], b[~y], facecolor="none", edgecolors="red", label="-")
    plt.legend()
    plt.show()


class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2 * n_dim, n_obj=2, xl=np.concatenate([xl, xl]), xu=np.concatenate([xu, xu]), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        lower, upper = x.reshape((2, -1))

        y_hat = np.logical_and(X >= lower, X <= upper).all(axis=1)

        p, pp = y.sum(), y_hat.sum()
        tp = (y & y_hat).sum()

        recall = tp / p

        if pp == 0:
            precision = 0.0
        else:
            precision = tp / pp

        out["F"] = [-recall, - precision]


class MyRepair(Repair):

    def _do(self, problem, X, **kwargs):
        for k, x in enumerate(X):
            lower, upper = x.reshape((2, -1))
            X[k] = np.concatenate([np.minimum(lower, upper), np.maximum(lower, upper)])

        return X


problem = MyProblem()

algorithm = NSGA2(repair=MyRepair())

res = minimize(problem,
               algorithm,
               ("n_gen", 150),
               verbose=True)

sols = res.opt

sols = sols[(-sols.get("F")[:, 0]).argsort()]

X = sols.get("X")
F = sols.get("F") * [-1, -1]

for k in range(len(sols)):
    lower, upper = X[k].reshape((2, -1))
    recall, precision = F[k]

    fig, axs = plt.subplots(2)
    top, bottom = axs

    fig.suptitle(f"recall: {recall:.4f} | precision: {precision:.4f}")

    top.scatter(F[:, 0], F[:, 1], facecolor="none", edgecolors="black")
    top.scatter(F[k, 0], F[k, 1], color="red", marker="x", s=50)
    top.set_xlabel("RECALL")
    top.set_ylabel("PRECISION")

    bottom.scatter(a[y], b[y], facecolor="none", edgecolors="green", label="+")
    bottom.scatter(a[~y], b[~y], facecolor="none", edgecolors="red", label="-")
    rect = patches.Rectangle(lower, *(upper - lower), linewidth=2, edgecolor='black', facecolor='none')
    bottom.add_patch(rect)

    plt.legend()
    plt.tight_layout()
    plt.show()
