import os
import numpy as np

from util.hypervolume import calc_hypervolume
from util.misc import get_f
import pandas as pd


def load_from_out(folder):
    files = os.listdir(folder)

    data = []

    for fname in files:

        if not fname.endswith('out'):
            continue

        array = fname[0:-4].split('_')
        gen = array[3] if len(array) == 4 else -1
        pop = np.loadtxt(os.path.join(folder, fname))

        r = np.array([1.01, 1.01])
        hv = calc_hypervolume(pop, r)

        d = {'algorithm': array[0],
             'problem': array[1],
             'run': array[2],
             'gen' : gen,
             #'pop' : pop,
             'hv' : hv
             }

        data.append(d)

        if len(data) % 200 == 0:
            print len(data) * 100 / float(len(files)), '%'

    return pd.DataFrame(data)


if __name__ == '__main__':
    df = load_from_out('../../results')

    print df

    f = {'hv': ['median', 'min', 'mean', 'max', 'std']}

    print df.groupby(['problem', 'algorithm']).agg(f).T

    #print df[df['hv'] > 1.0]



    # np.save('out', data, allow_pickle=True)
    # data = np.load('out.npy', allow_pickle=True)
    # print len(data)
