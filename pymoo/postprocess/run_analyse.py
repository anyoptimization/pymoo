import os
import re
import sys

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from plotly.graph_objs import Layout
from pymeta.configuration import Configuration
# this import is needed for the reflection object to get the true front -> don't remove it
from pymoo.problems.ZDT.zdt1 import ZDT1
from pymoo.problems.ZDT.zdt2 import ZDT2
from pymoo.problems.ZDT.zdt3 import ZDT3
from pymoo.problems.ZDT.zdt4 import ZDT4

from pymoo.performance.hypervolume import Hypervolume
from pymoo.performance.igd import IGD
from pymoo.util.non_dominated_rank import NonDominatedRank


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

    data = load_files(folder, ".*_ZDT.*_.*\.out", ["algorithm", "problem", "run"])

    for entry in data:

        pop = np.loadtxt(entry['path'])

        if len(np.shape(pop)) == 1:
            pop = np.array([pop])
        else:
            pop = pop[NonDominatedRank.calc_as_fronts(pop, None, only_pareto_front=True)]

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
            print("True front for problem %s not found. Can't calculate IDG. Continue." % entry['problem'])
            continue


        print(reference_point)
        entry['igd'] = IGD(true_front).calc(pop)
        entry['hv'] = Hypervolume(reference_point).calc(pop)

        #print(reference_point)

        del entry['path']
        del entry['fname']

    return pd.DataFrame(data)






def load_files(folder, regex, columns=[], split_char="_"):
    data = []
    for fname in os.listdir(folder):

        if not re.match(regex, fname):
            continue

        sname = fname.split(".")[0]
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
    df = load(Configuration.BENCHMARK_DIR + "standard")

    with pd.option_context('display.max_rows', None):
        print(df)
    #    print(df[(df.problem == 'ZDT3') & (df.igd > 0.03)])

    with pd.option_context('display.max_rows', None):
        f = {'igd': ['median', 'min', 'mean', 'max', 'std']}
        print(df.groupby(['problem', 'algorithm']).agg(f))

    problems = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]

    for problem in problems:

        data = df[(df.problem == problem)]

        F = []
        for algorithm in data.algorithm.unique():
            F.append(np.array(data[data.algorithm == algorithm].igd.tolist()))

        create_plot("%s.html" % problem, "Measure %s" % problem, F, chart_type="box", labels=data.algorithm.unique())




