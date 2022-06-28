import numpy as np

from pymoo.algorithms.moo.rvea import APDSurvival, RVEA

from pymoo.core.population import Population
from pymoo.problems.many import DTLZ2
from tests.test_util import load_to_test_resource


def test_survival():
    problem = DTLZ2(n_obj=3)

    for k in range(1, 11):
        print("TEST RVEA GEN", k)

        ref_dirs = np.loadtxt(load_to_test_resource('rvea', f"ref_dirs_{k}.txt"))
        F = np.loadtxt(load_to_test_resource('rvea', f"F_{k}.txt"))
        pop = Population.new(F=F)

        algorithm = RVEA(ref_dirs)
        algorithm.setup(problem, termination=('n_gen', 500))
        algorithm.n_gen = k
        algorithm.pop = pop

        algorithm.termination.update(algorithm)

        survival = APDSurvival(ref_dirs)
        survivors = survival.do(problem, algorithm.pop, n_survive=len(pop), algorithm=algorithm, return_indices=True)

        apd = pop[survivors].get("apd")
        correct_apd = np.loadtxt(load_to_test_resource('rvea', f"apd_{k}.txt"))
        np.testing.assert_allclose(apd, correct_apd)
