import argparse
import pickle

import numpy as np

from algorithms.naive import NaiveMetamodelAlgorithm
from algorithms.nsao import NSAO
from algorithms.nsga import NSGA
from model.evaluator import Evaluator
from problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from rand.my_random_generator import MyRandomGenerator
from util.misc import uniform_2d_weights


def save_hist(pathToFile, data):
    hist = None
    for i in range(len(data)):
        obj = data[i]['snapshot']
        hist = obj if hist is None else np.concatenate((hist, obj), axis=0)

    np.savetxt(pathToFile, hist, fmt='%.14f')
    print(pathToFile)


def save_dat(pathToFile, data):
    with open(pathToFile, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(pathToFile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n', metavar='n', default=None, type=str, help='Execute a specific run (HPCC)')
    parser.add_argument('-o', '--out', default=None, help='Defines the storage of the output file(s).')
    parser.add_argument('--hist', help='If true a .hist file with the pareto fronts over time is created.',
                        action='store_true')
    parser.add_argument('--dat', help='Additional data saved to the algorithm - may differ for each algorithm - are '
                                      'stored.',
                        action='store_true')
    args = parser.parse_args()

    type = 'expensive'
    runs = 30
    rng = MyRandomGenerator()

    if type == 'standard':
        pop_size = 100
        name = "pynsga-myrandom"
        algorithm = NSGA(pop_size=100, )
        problems = [ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6()]
        n_gen = [200, 200, 200, 200, 400]

    elif type == 'expensive':
        algorithm = NSAO(50, reference_directions=uniform_2d_weights(6))
        name = "mma-gpflow-mean"
        #algorithm = NaiveMetamodelAlgorithm(50)
        n_eval = 100
        problems = [ZDT1(n_var=5), ZDT2(n_var=5), ZDT3(n_var=5)]

    counter = 0
    n = [int(args.n)] if '-' not in args.n else range(int(args.n.split('-')[0]), int(args.n.split('-')[1]) + 1)

    for p in range(len(problems)):

        for r in range(runs):

            counter += 1
            if args.n is not None and counter not in n:
                continue

            evaluator = Evaluator(n_eval)
            x, f, g = algorithm.solve(problems[p], evaluator, seed=r, rnd=rng)

            if args.out is not None:

                sPathPrefix = args.out + '/%s_%s_%02d' % (name, problems[p].name(), (r + 1))

                np.savetxt(sPathPrefix + ".out", np.asarray(f), fmt='%.14f')
                print(sPathPrefix + ".out")

                if args.hist:
                    save_hist(sPathPrefix + ".hist", evaluator.data)

                if args.dat:
                    save_dat(sPathPrefix + ".dat", evaluator.data)
