import matplotlib.pyplot as plt
import numpy as np
import pandas
import plotly.plotly as py
import plotly.graph_objs as go
import pygmo
import scipy
import plotly
from pandas.plotting import parallel_coordinates
from plotly.graph_objs import Scatter, Layout
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from util.non_dominated_rank import NonDominatedRank

vars = np.array(['pctExh1', 'pctExh2', 'pctExh3', 'pctExh4', 'pctInt1', 'pctInt2', 'pctInt3', 'pctInt4'])
objectives = np.array(['min_htc', 'p_drop'])


def feature_importance(X, y, obj):
    tree = DecisionTreeRegressor(random_state=0).fit(X, y[:, obj])
    val = np.array(tree.feature_importances_)

    scoring = make_scorer(r2_score)
    scores = cross_val_score(tree, X, y[:, obj], scoring=scoring)
    print "mean: {:.3f} (std: {:.3f})".format(scores.mean(),scores.std())

    print objectives[obj]
    print np.array_str(np.array([vars, val]).T, precision=5, suppress_small=True)
    print '\n'


def show_correlation_plot(X, y, var, obj):
    plt.scatter(X[:, var], y[:, obj], alpha=0.5)
    plt.title('Correlation between variable %s and objective %s' % (var, obj))
    plt.xlabel('var %s' % vars[var])
    plt.ylabel('F %s' % objectives[obj])
    plt.show()


if __name__ == '__main__':

    problem = 'b38'
    data = np.loadtxt('/Users/julesy/workspace/%s.out' % problem, dtype=float)


    X = np.array(data[:, 1:-2])
    y = data[:, [-2, -1]]

    df = pandas.DataFrame(y)
    df['rank'] = NonDominatedRank.calc_from_fronts(pygmo.fast_non_dominated_sorting(y)[0])
    front = df[df['rank'] == 0]

    plotly.offline.plot({
        "data": [Scatter(x=df[df['rank'] != 0][0], y=df[df['rank'] != 0][1], mode = 'markers', name='DOE'), Scatter(x=df[df['rank'] == 0][0], y=df[df['rank'] == 0][1], mode = 'markers', name = 'front')],
        "layout": Layout(title="Objective Space %s" % problem)
    })




    epoch = data[:, 0]

    # print y


    print 'Correlation of Variables:'

    for i in range(len(objectives)):
        for j in range(len(vars)):
            corr = scipy.stats.pearsonr(X[:, j], y[:, i])
            print '%s [%s] - %s [%s]: %s' % (objectives[i], i, vars[j], j, corr)

    print '---------------------------'

    print 'Feature Importance:'
    feature_importance(X, y, 0)
    feature_importance(X, y, 1)
    feature_importance(X, y, [0, 1])

    print '---------------------------'




    df = pandas.DataFrame(X)
    df['rank'] = NonDominatedRank.calc_from_fronts(pygmo.fast_non_dominated_sorting(y)[0])
    parallel_coordinates(df[(df['rank']==0) | (df['rank']==1)], 'rank')
    plt.show()

    print df[(df['rank']==0)]

    print 'test'


    #show_correlation_plot(X, y, 0, 0)
    show_correlation_plot(X, y, 5, 0)
    #show_correlation_plot(X, y, 7, 0)
    #show_correlation_plot(X, y, 7, 1)
