import os
import sys
import time

import numpy as np
from pymop.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

from pymoo.algorithms.NSGAIII import NSGAIII
from pymoo.model.evaluator import Evaluator

n_runs = 20
output = os.path.join('..', '..', '..', 'benchmark', 'standard')
if not os.path.exists(output):
    output = os.path.join('.')


def get_params():
    for algorithm in [NSGAIII("real", verbose=0)]:
        for problem in [ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6()]:
            for run in range(1, n_runs + 1):
                yield (algorithm, problem, run)


if __name__ == '__main__':

    params = list(get_params())

    print("Parameters found: %s" % len(params))

    if len(sys.argv) == 1:
        print("Please provide a parameter settings value.")
        exit(1)

    start = int(sys.argv[1]) - 1
    end = start+1
    if len(sys.argv) == 3:
        end = int(sys.argv[2]) - 1

    print("Run for parameter settings: %s - %s" % (start, end))

    for i in range(start,end):

        algorithm, problem, run = params[i]

        eval = Evaluator(20000)
        start_time = time.time()
        X, F, G = algorithm.solve(problem, evaluator=eval, seed=run)
        print("--- %s seconds ---" % (time.time() - start_time))

        # save the result as a test
        fname = os.path.join(output, 'pynsga3_' + problem.__class__.__name__ + '_%s' % run)
        np.savetxt(fname + ".out", F)
        print(fname + ".out")
        #save_hist(fname + ".hist", eval.data)
