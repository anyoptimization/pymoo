import argparse

import numpy as np

from algorithms.nsga import NSGA
from model.evaluator import Evaluator
from operators.random_factory import RandomFactory
from operators.random_spare_factory import RandomSpareFactory
from problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from rand.default_random_generator import DefaultRandomGenerator
from rand.my_random_generator import MyRandomGenerator
from rand.numpy_random_generator import NumpyRandomGenerator
from util.misc import get_f
from rand.secure_random_generator import SecureRandomGenerator

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n', metavar='n', default=None, type=str, help='Execute a specific run (HPCC)')
    parser.add_argument('-o', '--out', default=None, help='Defines the storage of the output file(s).')
    parser.add_argument('--hist', help='If true a .hist file with the pareto fronts over time is created.',
                        action='store_true')
    args = parser.parse_args()

    pop_size = 100
    name = "pynsga-myrandom"

    algorithm = NSGA(pop_size=100, )
    problems = [ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6()]
    n_gen = [200, 200, 200, 200, 400]
    runs = 30
    rng = MyRandomGenerator()

    counter = 0

    n = [int(args.n)] if '-' not in args.n else range(int(args.n.split('-')[0]), int(args.n.split('-')[1]) + 1)

    for p in range(len(problems)):

        for r in range(runs):

            counter += 1
            if args.n is not None and counter not in n:
                continue

            evaluator = Evaluator(n_gen[p] * pop_size)
            pop = algorithm.solve(problems[p], evaluator, seed=r, rnd=rng)

            if args.out is not None:
                data = evaluator.data

                # save the final population
                fname = args.out + '/%s_%s_%02d.out' % (name, problems[p].name(), (r + 1))
                print(fname)
                np.savetxt(fname, np.asarray(get_f(pop)), fmt='%.14f')

                # save the files for each generation
                hist = None

                if args.hist:

                    for i in range(len(data)):
                        obj = np.asarray(get_f(data[i]))
                        obj = np.insert(obj, 0, i * np.ones(len(obj)), axis=1)
                        hist = obj if hist is None else np.concatenate((hist, obj), axis=0)

                    fname = args.out + '/%s_%s_%02d.hist' % (name, problems[p].name(), (r + 1))
                    print(fname)
                    np.savetxt(fname, hist, fmt='%.14f')
