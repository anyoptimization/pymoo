import numpy as np
import pandas
import os
from configuration import Configuration
from metamodels.selection_error_probablity import calc_sep
from util.misc import calc_mse, load_files, create_plot, calc_r2, calc_rmse, normalize, calc_amse

if __name__ == '__main__':

    folder = os.path.join(Configuration.BENCHMARK_DIR, "metamodels/")
    data = load_files(folder, ".*.*\.out", ["model", "problem", "run"])

    measure = "amse"

    data = [e for e in data if "ford" in e["fname"]]

    for entry in data:


        F_test_fname = folder + entry['problem'] + '_' + entry['run'] + ".f_test"
        F = np.loadtxt(F_test_fname)
        F_hat = np.loadtxt(entry['path'])

        F_min = np.min(F)
        F_max = np.max(F)


        if measure == "mse":
            entry['mse'] = calc_mse(F, F_hat)
        if measure == "sep":
            entry['sep'] = calc_sep(F, F_hat)
        if measure == "r2":
            entry['r2'] = calc_r2(F, F_hat)
        if measure == "rmse":
            entry['rmse'] = calc_rmse(normalize(F,F_min, F_max), normalize(F_hat,F_min, F_max))
        if measure == "amse":
            entry['amse'] = calc_amse(F, F_hat)


        del entry['path']
        del entry['fname']

        print(F_test_fname)

    df = pandas.DataFrame(data)

    with pandas.option_context('display.max_rows', None):
        print(df)
        print(df.groupby(['model']).agg({measure: ['mean', 'std']}))
        print(df.groupby(['model', 'problem']).agg({measure: ['mean']}))

    X = []
    F = []

    for model in df.model.unique():
        data = df[df.model == model]
        F.append(np.array(data[measure].tolist()))
        X.append(np.array(data.problem.tolist()))

    create_plot("models_bar_.html", "Metamodel Performance", F, X=X, chart_type="box", labels=df.model.unique(),
                grouped=True)
