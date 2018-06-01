import os
import sys
import time

import numpy as np
from pymop.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from pymop.dtlz import DTLZ2

from pymoo.algorithms.RNSGAIII import RNSGAIII
from pymoo.model.evaluator import Evaluator
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover

n_runs = 30
output = os.path.join('..', 'benchmark', 'RNSGAII_Benchmark')
if not os.path.exists(output):
    output = os.path.join('.')


def ZDT_Test():
    ref_dirs = {ZDT1(): [[(0.2, 0.4), (0.8, 0.4)],
                         [(0.2, 0.6), (0.4, 0.6), (0.5, 0.2), (0.7, 0.2), (0.9, 0)]],
                ZDT2(): [[(0.2, 0.8), (0.7, 1), (0.8, 0.2)]],
                ZDT3(): [[(0.1, 0.6), (0.3, 0.2), (0.7, -0.25)]]}
    crossover = SimulatedBinaryCrossover(10)

    for problem, ref_points in ref_dirs.items():
        for points in ref_points:
            for algorithm in [RNSGAIII("real", pop_size=100, ep=0.001, crossover=crossover, ref_dirs=points, verbose=0)]:
                for run in range(1, n_runs + 1):
                    yield (algorithm, problem, run)

def DTLZ_test():
    ref_dirs = {DTLZ2: [{"n_var": 11, "n_obj": 3, "ref_points": [[[0.2, 0.2, 0.6], [0.8, 0.8, 0.8]]]},
                          {"n_var": 14, "n_obj": 5, "ref_points": [[[0.5, 0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.2, 0.2, 0.8]]]},
                          {"n_var": 19, "n_obj": 10, "ref_points": [[[0.25 for i in range(10)]]]}
                          ]
                }
    crossover = SimulatedBinaryCrossover(10)

    for problem, setup in ref_dirs.items():
        for params in setup:
            parameters = {"n_var": params["n_var"], "n_obj": params["n_obj"]}
            print(problem)
            prob = problem(**parameters)
            for points in params["ref_points"]:
                for algorithm in [RNSGAIII("real", pop_size=100, ep=0.01, crossover=crossover, ref_dirs=points, verbose=0)]:
                    for run in range(1, n_runs + 1):
                        yield (algorithm, prob, run)


if __name__ == '__main__':

    params = list(ZDT_Test()) + list(DTLZ_test())
    total_time = time.time()
    for i in range(len(params)):

        algorithm, problem, run = params[i]
        n_gen = 500
        eval = Evaluator(100*n_gen)
        start_time = time.time()
        X, F, G = algorithm.solve(problem, evaluator=eval, seed=run)
        print("--- %s seconds ---" % (time.time() - start_time))

        # save the result as a test
        fname = os.path.join(output, 'pynsga3_' + problem.__class__.__name__ + '_%s' % run)
        np.savetxt(fname + ".out", F)
        print(fname + ".out")
        # save_hist(fname + ".hist", eval.data)
    print(f"Benchmark took {time.time()-total_time} seconds")