import os

import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.bin_random_sampling import BinaryRandomSampling
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowdingSurvival
from pymoo.problems.ZDT.zdt1 import ZDT1
from pymoo.problems.knapsack import Knapsack


def write_final_pop_obj(pop, run):
    f_name = os.path.join('results', problem.__class__.__name__ + '_RUN' + str(run) + str('.out'))
    f = open(f_name, 'w')
    for ind in pop:
        f.write('%f \t %f\n' % (ind.f[0], ind.f[1]))
    f.close()


if __name__ == '__main__':

    problem = ZDT1()

    n_items = 20
    P = np.random.randint(100, size=n_items)
    W = np.random.randint(100, size=n_items)
    C = np.sum(W) / 2
    problem = Knapsack(n_items, W, P, C)
    problem.evaluate(BinaryRandomSampling().sample(problem, 3))

    problem = ZDT1()

    # print(problem)

    import time

    start_time = time.time()
    X, F, G = GeneticAlgorithm(100,
                               sampling=RealRandomSampling(),
                               selection=TournamentSelection(),
                               crossover=SimulatedBinaryCrossover(),
                               mutation=PolynomialMutation(p_mut=0.033),
                               survival=RankAndCrowdingSurvival(),
                               verbose=1
                               ).solve(problem, evaluator=20000, seed=12345)
    print("--- %s seconds ---" % (time.time() - start_time))

    fname = os.path.join('..', '..', '..', 'benchmark', 'standard', 'pynsganew_' + problem.__class__.__name__ + '_1' + str('.out'))
    np.savetxt(fname, F)

    plt.scatter(F[:,0], F[:,1])
    plt.show()

    # r = np.array([1.01, 1.01])

    # print(calc_hypervolume(get_f(pop), r))

    # write_final_pop_obj(pop,1)
