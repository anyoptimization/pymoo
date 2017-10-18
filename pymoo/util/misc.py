import re

import numpy as np
import pygmo
import os
import plotly
from plotly.graph_objs import Layout
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, r2_score
from configuration import Configuration
from operators.random_factory import RandomFactory


def get_x(pop):
    return np.array([pop[i].x for i in range(len(pop))])


def get_f(pop):
    return np.array([pop[i].f for i in range(len(pop))])


def get_g(pop):
    return np.array([pop[i].g for i in range(len(pop))])


def evaluate(evaluator, problem, pop):
    f, g = evaluator.eval(problem, np.array([ind.x for ind in pop]))

    for i in range(len(pop)):
        pop[i].f = f[i, :]
        pop[i].g = g[i, :]


def denormalize(x, min, max):
    return x * (max - min) + min


def normalize(x, min, max):
    return (x - min) / (max - min)


def print_pop(pop, rank, crowding, sorted_idx, n):
    for i in range(n):
        print(i, pop[sorted_idx[i]].f, rank[sorted_idx[i]], crowding[sorted_idx[i]])
    print('---------')


def perpendicular_dist(ref_dir, point):
    projection = (np.dot(point, ref_dir) / np.linalg.norm(ref_dir)) * ref_dir
    return np.linalg.norm(projection - point)


def calc_mse(f, f_hat):
    return mean_squared_error(f, f_hat)
    #return np.sum(np.power(f_hat - f, 2), axis=0)

def calc_r2(f, f_hat):
    return r2_score(f, f_hat)

def calc_rmse(f, f_hat):
    return np.sqrt(((f_hat - f) ** 2).mean())

def calc_amse(f, f_hat):
    return np.mean(np.abs( (f - f_hat) / f) )


def uniform_2d_weights(n_weights):
    v = np.linspace(0, 1.0, n_weights)
    return np.array([v, 1 - v]).T


def calc_metamodel_goodness(problem, metamodel, n=100, X=None, func=calc_mse):
    if X is None:
        X = RandomFactory().sample(n, problem.xl, problem.xu)
    f, _ = problem.evaluate(X)
    f_hat, std = metamodel.predict(X)
    return np.array([func(f[:, i], f_hat[:, i]) for i in range(problem.n_obj)])


def get_front_by_index(f):
    return pygmo.fast_non_dominated_sorting(f)[0][0]


def get_front(f):
    return f[get_front_by_index(f)]


def get_hist_from_pop(pop, n_evals):
    return np.concatenate((np.ones((len(pop), 1)) * n_evals, get_x(pop), get_f(pop), get_g(pop)), axis=1)


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


def create_plot(fname, title, F, X=None, chart_type="line", labels=[], folder=Configuration.BENCHMARK_DIR, grouped=False):
    plots = []
    for m in range(len(F)):
        if m < len(labels):
            label = labels[m]
        else:
            label = m

        if type(F) is list:
            data = F[m]
        elif type(F) is np.ndarray:
            data = F[m,:]

        if type(X) is list:
            if len(X) == 1:
                data_X = X
            else:
                data_X = X[m]
        elif type(X) is np.ndarray:
            data_X = X[m,:]

        if chart_type is "line":
            if X is None:
                X = np.array(list(range(len(F.shape[0]))))
            plot = go.Scatter(x=data_X,y=data,mode='lines+markers',name=label)
        elif chart_type is "box":
            if grouped:
                plot = go.Box(x=data_X, y=data, name=label)
            else:
                plot = go.Box(y=data, name=label)

        elif chart_type is "bar":
            plot = go.Bar(x=data_X, y=data,name=label)

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

