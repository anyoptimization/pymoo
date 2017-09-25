import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from algorithms.nsga import NSGA
from problems.zdt import ZDT1
from util.hypervolume import calc_hypervolume
from util.misc import get_f


def write_final_pop_obj(pop, run):
    f_name = os.path.join('results', problem.__class__.__name__ + '_RUN' + str(run) + str('.out'))
    f = open(f_name, 'w')
    for ind in pop:
        f.write('%f \t %f\n' % (ind.f[0], ind.f[1]))
    f.close()


if __name__ == '__main__':
    print os.path.dirname(os.path.abspath(__file__))


    sys.path.extend('/Users/julesy/workspace/moo-python')

    problem = ZDT1()
    pop = NSGA(pop_size=8).solve(problem, evaluator=800, seed=4321)

    print np.array_str(np.asarray(get_f(pop)))

    x = [pop[i].f[0] for i in range(len(pop))]
    y = [pop[i].f[1] for i in range(len(pop))]
    plt.scatter(x, y)
    plt.show()

    r = np.array([1.01, 1.01])

    print calc_hypervolume(get_f(pop), r)

    # write_final_pop_obj(pop,1)
