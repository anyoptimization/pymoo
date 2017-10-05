import sklearn
from sklearn.gaussian_process import gaussian_process
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic, ExpSineSquared, WhiteKernel, Matern
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
import itertools

from metamodels.selection_error_probablity import selection_error_probability


class MetaModel:
    def __init__(self):
        self.parameters = None
        self.gp = []
        self.goodness = []

    def fit(self, X, F):

        kernels = [WhiteKernel(), RationalQuadratic(), RBF(), ConstantKernel(), Matern()]
        n_objectives = np.shape(F)[1]

        self.goodness = np.zeros((n_objectives, len(kernels)))

        for m in range(n_objectives):

            for kernel_idx in range(len(kernels)):
                kernel = kernels[kernel_idx]
                gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
                # do cross validation
                #scorer = make_scorer(selection_error_probability, greater_is_better=False)
                scores = cross_val_score(gp, X, F[:, m], cv=4, scoring='neg_mean_squared_error')
                # take the mean of the results
                self.goodness[m, kernel_idx] = -np.mean(scores)

            best = np.argmin(self.goodness[m])
            gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernels[best])
            gp.fit(X, F[:, m])
            self.gp.append(gp)

            print m
            print gp

    def predict(self, X):

        n = len(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            n = 1

        f = np.zeros((len(self.gp), n))
        for j in range(len(self.gp)):
            f[j, :], _ = self.gp[j].predict(X, return_cov=True)
        return np.transpose(f)
