import numpy as np
import pandas
import os
from configuration import Configuration
from util.misc import calc_mse, load_files, create_plot

if __name__ == '__main__':

    folder = os.path.join(Configuration.BENCHMARK_DIR, "metamodels/")
    data = load_files(folder, ".*_.*\.out", ["model", "problem", "run"])


    for entry in data:

        F = np.loadtxt(folder + entry['problem'] + '_' + entry['run'] + ".f_test")
        F_hat = np.loadtxt(entry['path'])

        entry['mse'] = calc_mse(F, F_hat)
        del entry['path']
        del entry['fname']


    df = pandas.DataFrame(data)

    with pandas.option_context('display.max_rows', None):
        print(df)
        print(df.groupby(['model']).agg({'mse': ['mean', 'std']}))
        print(df.groupby(['model', 'problem']).agg({'mse': ['mean']}))

    X = []
    F = []

    for model in df.model.unique():

        l = []
        for problem in df.problem.unique():
            val = np.array(df[(df.model == model) & (df.problem == problem)].mse.tolist())
            l.extend(val)
            if len(F) == 0:
                X.extend([problem] * len(val))
        F.append(l)



    create_plot("models_bar.html", "Metamodel Performance", F, X=X, chart_type="box", labels=df.model.unique(), grouped=True)





