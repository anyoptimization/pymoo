import numpy as np

from pymoo.interface import crossover
from pymoo.operators.crossover.parent_centric_crossover import PCX
from pymoo.visualization.scatter import Scatter

X = np.eye(3)
X[1] = [0.9, 0.1, 0.1]

n_points = 1000

a = X[[0]].repeat(n_points, axis=0)
b = X[[1]].repeat(n_points, axis=0)
c = X[[2]].repeat(n_points, axis=0)

obj = PCX(eta=0.1, zeta=0.1, impl="elementwise")

_X = crossover(obj, a, c, b, xl=-1, xu=1)
sc = Scatter()
sc.add(_X, facecolor=None, edgecolor="blue", alpha=0.7)
sc.add(X, s=100, color="red")
sc.add(X.mean(axis=0), color="green", label="Centroid")
sc.show()
