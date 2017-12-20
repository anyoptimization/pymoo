import os
import time

import matplotlib.pyplot as plt
import numpy as np

from pymoo.algorithms.NSGAII import NSGAII
from pymoo.model.evaluator import Evaluator
from pymoo.problems.ZDT.zdt1 import ZDT1
from pymoo.util.misc import save_hist

if __name__ == '__main__':

    problem = ZDT1()

    for run in range(1,11):



        start_time = time.time()

        # run the algorithm
        nsga = NSGAII("real", verbose=True)
        eval = Evaluator(20000)
        X, F, G = nsga.solve(problem, evaluator=eval, seed=12345)
        print("--- %s seconds ---" % (time.time() - start_time))


        # save the result as a test
        fname = os.path.join('..', '..', '..', 'benchmark', 'standard',
                             'pynsganew_' + problem.__class__.__name__ + '_%s' % run)
        np.savetxt(fname + ".out", F)
        save_hist(fname + ".hist", eval.data)

        # save the whole history

        plot = False
        if plot:
            plt.scatter(F[:, 0], F[:, 1])

            for l in nsga.ref_lines[1:-1]:
                x = np.linspace(0,1,100)
                y = x / l[1] * l[0]
                plt.plot(x,y)

            plt.ylim(0,1)
            plt.xlim(0,1)
            plt.show()

    # r = np.array([1.01, 1.01])

    # print(calc_hypervolume(get_f(pop), r))

    # write_final_pop_obj(pop,1)
