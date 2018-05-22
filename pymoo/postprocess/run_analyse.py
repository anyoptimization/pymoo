import os
import re
import sys

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from plotly.graph_objs import Layout
from pymoo.configuration import Configuration
from pymoo.performance.hypervolume import Hypervolume
from pymoo.performance.igd import IGD
from pymoo.util.non_dominated_rank import NonDominatedRank

# this import is needed for the reflection object to get the true front -> don't remove it
from pyop.problems.zdt import *
from pyop.problems.rastrigin import *
from pyop.problems.ackley import *

# this import is needed for the reflection object to get the true front -> don't remove it


def create_plot(fname, title, F, X=None, chart_type="line", labels=[], folder=Configuration.BENCHMARK_DIR,
                grouped=False):
    plots = []
    for m in range(len(F)):
        if m < len(labels):
            label = labels[m]
        else:
            label = m

        if type(F) is list:
            data = F[m]
        elif type(F) is np.ndarray:
            data = F[m, :]

        if type(X) is list:
            if len(X) == 1:
                data_X = X
            else:
                data_X = X[m]
        elif type(X) is np.ndarray:
            if X.ndim == 1:
                data_X = X
            else:
                data_X = X[m, :]

        if chart_type is "line":
            if X is None:
                X = np.array(list(range(len(F.shape[0]))))
            plot = go.Scatter(x=data_X, y=data, mode='lines+markers', name=label)
        elif chart_type is "box":
            if grouped:
                plot = go.Box(x=data_X, y=data, name=label)
            else:
                plot = go.Box(y=data, name=label)

        elif chart_type is "bar":
            plot = go.Bar(x=data_X, y=data, name=label)

        plots.append(plot)

    if folder is not None:
        fname = os.path.join(folder, fname)

    if grouped:
        layout = Layout(title=title, barmode='group', boxmode='group')
    else:
        layout = Layout(title=title)

    plotly.offline.plot(
        {
            "data": plots,
            "layout": layout
        },
        filename=fname
    )


def load(folder):
    data = load_files(folder, ".*_.*_.*\.out", ["algorithm", "problem", "run"])

    for entry in data:

        F = np.loadtxt(entry['path'])

        problem = globals()[entry['problem']]()

        if problem.n_obj == 1:
            entry['igd'] = np.min(F)
            entry['hv'] = np.min(F)
        else:

            if len(np.shape(F)) == 1:
                F = np.array([F])
            else:
                F = F[NonDominatedRank.calc_as_fronts(F, None, only_pareto_front=True)]

            true_front = None
            try:
                true_front = problem.pareto_front()
                reference_point = problem.nadir_point() * 1.01
            except:
                print("Unexpected error:", sys.exc_info()[0])

            if true_front is None:
                print("True front for problem %s not found. Can't calculate IDG. Continue." % entry['problem'])
                continue

            # print(reference_point)
            entry['igd'] = IGD(true_front).calc(F)
            entry['hv'] = Hypervolume(reference_point).calc(F)

        # print(reference_point)

        del entry['path']
        del entry['fname']

    return pd.DataFrame(data)


def load_files(folder, regex, columns=[], split_char="_"):
    data = []

    files = []
    [files.extend([os.path.join(dirpath, f) for f in filelist]) for dirpath, _, filelist in os.walk(folder)]

    for i, fname in enumerate(files):

        if not re.match(regex, fname.split("/")[-1]):
            continue

        sname = fname.split("/")[-1].split(".")[0]
        entry = {}

        array = sname.split(split_char)

        for i, column in enumerate(columns):
            if i < len(array):
                entry[column] = array[i]
            else:
                entry[column] = None

        entry['fname'] = fname
        entry['path'] = os.path.join(folder, fname)

        data.append(entry)
    return data


if __name__ == '__main__':
    df = load(Configuration.BENCHMARK_DIR + "expensive")
    df = df.sort_values(['algorithm', 'problem'])

    with pd.option_context('display.max_rows', None):
        print(df)
    # print(df[(df.problem == 'ZDT3') & (df.igd > 0.03)])

    with pd.option_context('display.max_rows', None):
        f = {'hv': ['median', 'min', 'mean', 'max', 'std']}
        print(df.groupby(['problem', 'algorithm']).agg(f))

    problems = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "Rastrigin", "Ackley"]

    for problem in problems:

        data = df[(df.problem == problem)]

        if data.size == 0:
            continue

        F = []
        for algorithm in data.algorithm.unique():
            F.append(np.array(data[data.algorithm == algorithm].igd.tolist()))

        create_plot("%s.html" % problem, "Measure %s" % problem, F, chart_type="box", labels=data.algorithm.unique())
