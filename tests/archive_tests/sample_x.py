import numpy as np

from pymop import *
from pymop.problems.pressure_vessel import PressureVessel

if __name__ == "__main__":

    n_samples = 100

    problems = [
        ("ctp1", CTP1()),
        ("ctp2", CTP2()),
        ("ctp3", CTP3()),
        ("ctp4", CTP4()),
        ("ctp5", CTP5()),
        ("ctp6", CTP6()),
        ("ctp7", CTP7()),
        ("ctp8", CTP8())
    ]

    for name, problem in problems:

        m = problem.n_var
        X = np.random.random((n_samples, m))
        for i in range(m):
            X[:, i] = X[:, i] * (problem.xu[i] - problem.xl[i]) + problem.xl[i]

        np.savetxt("%s.x" % name, X)
