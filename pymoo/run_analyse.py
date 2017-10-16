import os

import numpy as np
import pygmo as pg
import pandas as pd
import sys
import plotly

from plotly.graph_objs import Layout

from measures.hypervolume import Hypervolume
from measures.igd import IGD
import plotly.graph_objs as go

# this import is needed for the reflection object to get the true front -> don't remove it
from problems.zdt import *
from problems.dtlz import *
from util.misc import load_files, create_plot


def load(folder):

    data = load_files(folder, ".*_.*_.*\.out", ["algorithm", "problem", "run"])

    for entry in data:

        pop = np.loadtxt(entry['path'])

        if len(np.shape(pop)) == 1:
            pop = np.array([pop])
        else:
            pop = np.array([pop[i, :] for i in pg.fast_non_dominated_sorting(pop)[0][0]])

        # if only a 1d array
        if len(pop.shape) == 1:
            pop = np.array([pop])

        true_front = None
        try:
            problem_clazz = globals()[entry['problem']]()
            true_front = problem_clazz.pareto_front()
            reference_point = problem_clazz.nadir_point() * 1.01
        except:
            print("Unexpected error:", sys.exc_info()[0])

        if true_front is None:
            print("True front for problem %s not found. Can't calculate IDG. Continue." % problem)
            continue

        entry['igd'] = IGD(true_front).calc(pop)
        entry['hv'] = Hypervolume(reference_point).calc(pop)

        del entry['path']
        del entry['fname']

    return pd.DataFrame(data)


if __name__ == '__main__':
    df = load('../../results_expensive')

    with pd.option_context('display.max_rows', None):
        print(df)
    #    print(df[(df.problem == 'ZDT3') & (df.igd > 0.03)])

    with pd.option_context('display.max_rows', None):
        f = {'igd': ['median', 'min', 'mean', 'max', 'std']}
        print(df.groupby(['problem', 'algorithm']).agg(f))

    problems = ["ZDT1", "ZDT2", "ZDT3"]

    for problem in problems:

        data = df[(df.problem == problem)]

        F = []
        for algorithm in data.algorithm.unique():
            F.append(np.array(data[data.algorithm == algorithm].igd.tolist()))
        create_plot("%s.html" % problem, "Measure %s" % problem, F, chart_type="box", labels=data.algorithm.unique())
