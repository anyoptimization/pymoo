import numpy as np

from pymoo.dm.knee_finding import KneeFinding
from pymoo.util.plotting import plot

if __name__ == '__main__':

    points = np.loadtxt("zdt1.pf")

    kf = KneeFinding(penalize_extremes=True)

    _points = np.array([[0.0, 1.0],
                       [0.1, 0.9],
                       [0.3, 0.7],
                       [0.5, 0.5],
                       [0.6, 0.4],
                       [0.7, 0.3],
                       [0.9, 0.1],
                       [1.0, 0.0],
                       ])

    I = kf.do(points)

    plot(points, points[[I]])

    print(points[I])

