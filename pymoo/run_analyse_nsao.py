import pickle
import plotly
import numpy as np
from plotly.graph_objs import Scatter, Layout
import pygmo

from configuration import Configuration
from run_analyse import load_files
import plotly.graph_objs as go

from util.misc import create_plot


def show_value(evals, val, name):
    for m in range(val.shape[1]):
        plotly.offline.plot(
            {
                "data": [
                    go.Scatter(
                        x=evals,
                        y=val[:, m],
                        mode='lines+markers',
                        name='lines+markers'
                    )
                ],
                "layout": Layout(title="%s Objective %s" % (name, (m + 1)))
            },
            filename='../../../%s_%s.html' % (name, (m + 1))
        )


def show_objective_space(n_evals, pop_f, f_hat, f_selected, f_selected_hat):
    # plot the current situation
    plots = []
    pop_non_dom = pop_f[pygmo.fast_non_dominated_sorting(pop_f)[0][0], :]
    plots.append(Scatter(x=pop_non_dom[:, 0], y=pop_non_dom[:, 1], name="True", mode='markers',
                         marker={"size": 8}))

    plots.append(Scatter(x=f_hat[:, 0], y=f_hat[:, 1], name="Prediction",
                         mode='markers',
                         marker={"size": 8, "symbol": "circle-open"}))

    plots.append(Scatter(x=f_selected_hat[:, 0], y=f_selected_hat[:, 1], name="Selected",
                         mode='markers',
                         marker={"size": 12}))

    for i in range(len(f_selected_hat)):
        plots.append(Scatter(x=np.array([f_selected_hat[i, 0], f_selected[i, 0]]),
                             y=np.array([f_selected_hat[i, 1], f_selected[i, 1]]),
                             name="Selected",
                             mode='lines+markers',
                             marker={"size": 4}))

    # merge the prediction and true values
    f = np.concatenate((pop_non_dom, f_hat))

    # normalize the values
    f_min = np.min(f, axis=0)
    f_max = np.max(f, axis=0)
    # f_norm = (f - f_min) / (f_max - f_min)
    # f_min = self.problem.ideal_point()
    # f_max = self.problem.nadir_point()

    # for ref in self.reference_directions:
    # denorm_x = ref[0] * (f_max[0] - f_min[0]) + f_min[0]
    # denorm_y = ref[1] * (f_max[1] - f_min[1]) + f_min[1]
    # if False:
    # plots.append(Scatter(x=np.array([f_min[0], denorm_x]),
    #                     y=np.array([f_min[1], denorm_y]),
    #                     name="Selected",
    #                     mode='lines+markers',
    #                     marker={"size": 4}))

    plotly.offline.plot(
        {
            "data": plots,
            "layout": Layout(title="Evaluations = %s" % n_evals)
        },
        filename=Configuration.BENCHMARK_DIR + 'run_%s.html' % n_evals
    )


if __name__ == '__main__':

    files = load_files(Configuration.BENCHMARK_DIR + "expensive", 'mma-gpflow-mean_ZDT3_01.dat')

    for entry in files:
        with open(entry['path'], 'rb') as handle:
            data = pickle.load(handle)
            for e in data:
                show_objective_space(e['n_evals'], e['pop_f'], e['f_hat'], e['f_selected'], e['f_selected_hat'])
                print(e['metamodels'])

            evals = np.array([e['n_evals'] for e in data])
            mse_true = np.array([e['mse_true'] for e in data])

            create_plot("mse.html", "MSE %s" % entry["fname"], mse_true.T, X=evals, chart_type="line")

            sep_true = np.array([e['sep_true'] for e in data])
            sep_true = np.array([e['sep_opt'] for e in data])

            print("Done")
