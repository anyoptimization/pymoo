import numpy as np
import plotly
from plotly.graph_objs import Layout, Scatter, Scatter3d

# this import is needed for the reflection object to get the true front -> don't remove it
from pymoo.configuration import Configuration
from pymoo.util.non_dominated_rank import NonDominatedRank


def load_non_dominated_from_file(file, non_dom=True):
    f = np.loadtxt(file)
    if non_dom:
        f = f[NonDominatedRank.get_front(f), :]
    return f


def parse_result_name(str):
    return str.split('/')[-1][0:-4]


def create_plot(file):
    f = load_non_dominated_from_file(file, True)
    if f.shape[1] == 2:
        return Scatter(x=f[:, 0], y=f[:, 1], name=parse_result_name(fn1), mode='markers', marker={"size": 8})
    elif f.shape[1] == 3:
        return Scatter3d(x=f[:, 0], y=f[:, 1], z=f[:, 2], name=parse_result_name(fn1), mode='markers',
                         marker={"size": 4})


plots = []

fn1 = Configuration.BENCHMARK_DIR + "standard/" + "LMA_TNK_0.out"
fn2 = Configuration.BENCHMARK_DIR + "expensive/" + "nsa-ea/nsa-ea_ZDT4_50.out"

plot = 1
compare = 0
plot_true_front = 0

problem = fn1.split('/')[-1][0:-4].split("_")[1]

if plot_true_front:
    true_front = globals()[problem]().pareto_front()
    plots.append(Scatter(x=true_front[:, 0], y=true_front[:, 1], mode='markers', name="true front",
                         marker={"size": 6, "symbol": "circle-open"}))

# if it should be compared and if it is the same problem
if compare:
    plots.append(create_plot(fn2))

if plot:
    plots.append(create_plot(fn1))

plotly.offline.plot({
    "data": plots,
    "layout": Layout(title="Objective Space",
                     xaxis=dict(title='Obj 1'),
                     yaxis=dict(title='Obj 2')
                     )
})
