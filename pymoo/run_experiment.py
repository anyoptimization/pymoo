import argparse

import numpy as np

from algorithms.nsga.nsga import NSGA
from model.evaluator import Evaluator
from problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from util.misc import get_f

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--n', metavar='n', default=None, type=int, help='Execute a specific run (HPCC)')
    parser.add_argument('--out', default=None, help='Defines the storage of the output file(s).')
    parser.add_argument('--gen', type=bool, default=False, help='If true each generation is written to a file.')
    args = parser.parse_args()

    algorithm = NSGA()
    problems = [ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6()]
    runs = 30

    counter = -1

    for p in problems:
        for r in range(runs):

            counter += 1
            if args.n is not None and counter != args.n:
                continue

            evaluator = Evaluator(20000)
            pop = algorithm.solve(p, evaluator, seed=r)

            if args.out is not None:
                data = evaluator.data

                # save the final population
                fname = args.out + '/%s_%s_%s.out' % ('nsga', p.name(), r)
                np.savetxt(fname, np.asarray(get_f(pop)))

                # save the files for each generation
                if args.gen:
                    for i in range(len(data)):
                        obj = np.asarray(get_f(data[i]))
                        fname = args.out + '/%s_%s_%s_%s.out' % ('nsga', p.name(), r, i)
                        print fname
                        np.savetxt(fname, obj)
