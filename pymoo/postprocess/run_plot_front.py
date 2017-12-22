import plotly
import pygmo as pg
from plotly.graph_objs import Layout, Scatter
from pymeta.configuration import Configuration
import numpy as np

# this import is needed for the reflection object to get the true front -> don't remove it


def load_non_dominated_from_file(file, non_dom = True):
    f = np.loadtxt(file)
    if non_dom:
        f = np.array([f[i, :] for i in pg.fast_non_dominated_sorting(f)[0][0]])
    return f[:, 0], f[:, 1]


def parse_result_name(str):
    return str.split('/')[-1][0:-4]


plots = []

fn1 = Configuration.BENCHMARK_DIR + "standard/" + "pynsga2_ZDT3_2.out"
fn2 = Configuration.BENCHMARK_DIR + "standard/" + "cnsga-rank_ZDT3_11.out"

plot = 1
compare = 1
plot_true_front = 0

problem = fn1.split('/')[-1][0:-4].split("_")[1]

if plot_true_front:
    true_front = globals()[problem]().pareto_front()
    plots.append(Scatter(x=true_front[:, 0], y=true_front[:, 1], mode='markers', name="true front",
                         marker={"size": 6, "symbol": "circle-open"}))

# if it should be compared and if it is the same problem
if compare:
    f2_x, f2_y = load_non_dominated_from_file(fn2, False)
    plots.append(Scatter(x=f2_x, y=f2_y, name=parse_result_name(fn2), mode='markers', marker={"size": 8}))

if plot:
    f_x, f_y = load_non_dominated_from_file(fn1, True)
    plots.append(Scatter(x=f_x, y=f_y, name=parse_result_name(fn1), mode='markers', marker={"size": 8}))

plotly.offline.plot({
    "data": plots,
    "layout": Layout(title="Objective Space",
                     xaxis=dict(title='Obj 1'),
                     yaxis=dict(title='Obj 2')
                     )
})
