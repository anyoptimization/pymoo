import os

import numpy as np
import pandas as pd
import sys

from measures.hypervolume import Hypervolume
from measures.igd import IGD

# this import is needed for the reflection object to get the true front -> don't remove it
from problems.zdt import *
from problems.dtlz import *


def load_from_out(folder):
    files = os.listdir(folder)

    data = []

    for fname in files:

        if not fname.endswith('out'):
            continue

        print("Parsing:  " + fname)
        array = fname[0:-4].split('_')

        algorithm = array[0]
        problem = array[1]
        run = array[2]
        gen = array[3] if len(array) == 4 else -1

        pop = np.loadtxt(os.path.join(folder, fname))

        # if only a 1d array
        if len(pop.shape) == 1:
            pop = np.array([pop])


        true_front = None
        try:
            problem_clazz = globals()[array[1]]()
            true_front = problem_clazz.pareto_front()
            reference_point = problem_clazz.nadir_point() * 1.01
        except:
            print("Unexpected error:", sys.exc_info()[0])

        if true_front is None:
            print("True front for problem %s not found. Can't calculate IDG. Continue." % problem)
            continue

        igd = IGD(true_front).calc(pop)
        hv = Hypervolume(reference_point).calc(pop)

        d = {'algorithm': algorithm,
             'problem': problem,
             'run': run,
             'gen': gen,
             # 'pop' : pop,
             'igd': igd,
             'hv' : hv,
             #'file' : fname
             }

        data.append(d)

        if len(data) % 200 == 0:
            print(len(data) * 100 / float(len(files)), '%')

    return pd.DataFrame(data)


if __name__ == '__main__':
    df = load_from_out('../../results')



    with pd.option_context('display.max_rows', None):
        print(df)
        print(df[(df.problem == 'ZDT3') & (df.igd > 0.03)])

    with pd.option_context('display.max_rows', None):
        f = {'igd': ['median', 'min', 'mean', 'max', 'std']}
        print(df.groupby(['problem', 'algorithm']).agg(f))


    # np.save('out', data, allow_pickle=True)
    # data = np.load('out.npy', allow_pickle=True)
    # print len(data)
