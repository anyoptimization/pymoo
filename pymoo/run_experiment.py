import argparse

import numpy as np

from algorithms.nsga import NSGA
from model.evaluator import Evaluator
from problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from util.misc import get_f

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n', metavar='n', default=None, type=int, help='Execute a specific run (HPCC)')
    parser.add_argument('-o', '--out', default=None, help='Defines the storage of the output file(s).')
    parser.add_argument('--hist', help='If true a .hist file with the pareto fronts over time is created.',
                        action='store_true')
    args = parser.parse_args()

    pop_size = 88

    algorithm = NSGA(pop_size=88)
    problems = [ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6()]
    n_gen = [100, 200, 200, 500, 1000]
    runs = 30

    counter = -1

    for p in range(len(problems)):

        for r in range(runs):

            counter += 1
            if args.n is not None and counter != args.n:
                continue

            evaluator = Evaluator(n_gen[p] * pop_size)
            pop = algorithm.solve(problems[p], evaluator, seed=r)

            if args.out is not None:
                data = evaluator.data

                # save the final population
                fname = args.out + '/%s_%s_%02d.out' % ('pynsga', problems[p].name(), (r + 1))
                print fname
                np.savetxt(fname, np.asarray(get_f(pop)), fmt='%.14f')

                # save the files for each generation
                hist = None

                if args.hist:

                    for i in range(len(data)):
                        obj = np.asarray(get_f(data[i]))
                        obj = np.insert(obj, 0, i * np.ones(len(obj)), axis=1)
                        hist = obj if hist is None else np.concatenate((hist, obj), axis=0)

                    fname = args.out + '/%s_%s_%02d.hist' % ('pynsga', problems[p].name(), (r + 1))
                    print fname
                    np.savetxt(fname, hist, fmt='%.14f')
