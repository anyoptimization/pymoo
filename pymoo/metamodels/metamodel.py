from random import shuffle

import numpy as np

from util.misc import calc_mse


class MetaModel:
    def __init__(self):
        self.parameters = []
        self.gp = []
        self.goodness = []
        self.f_min = None
        self.f_max = None
        self.x_min = None
        self.x_max = None

    def _get_parameter(self, d={}):
        pass

    def _predict(self, metamodel, X):
        pass

    def _create_and_fit(self, parameter, X, F, expensive=False):
        pass

    def fit(self, X, F):

        # only unique rows for X
        y = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
        _, idx = np.unique(y, return_index=True)
        X = X[idx]
        F = F[idx]

        self.f_min = np.min(F, axis=0)
        self.f_max = np.max(F, axis=0)
        F_norm = (F - self.f_min) / (self.f_max - self.f_min)

        self.x_min = np.min(X, axis=0)
        self.x_max = np.max(X, axis=0)
        X_norm = (X - self.x_min) / (self.x_max - self.x_min)

        #F_norm = F
        #X_norm = X

        n_observations = np.shape(F)[0]
        n_objectives = np.shape(F)[1]

        self.goodness = []

        # some metamodels need more information - it depends on the implementation
        d = {
            'n_var': X.shape[1]
        }

        # for each objective
        for m in range(n_objectives):

            # for each metamodel
            parameter = self._get_parameter(d)

            if len(parameter) == 1:
                mean_scores = [0.0]
            else:
                mean_scores = []
                for par in parameter:

                    scores = []
                    for train, test in self.k_fold_cross_validation(range(n_observations), 3, False):
                        # fit the meta model and predict results
                        model = self._create_and_fit(par, X_norm[train, :], F_norm[train, m])
                        f_hat, _ = self._predict(model, X_norm[test, :])
                        f_true = F_norm[test, m]

                        # calc metric for this run
                        metric = calc_mse(f_true, f_hat)
                        scores.append(metric)

                    # save the goodness for this metamodel
                    mean_scores.append(np.mean(scores))

            # get the best metamodel for this objective
            best_i = np.argmin(mean_scores)

            # train it on all training data
            par = self._get_parameter(d)[best_i]
            print(par)
            self.parameters.append(par)

            best = self._create_and_fit(par, X_norm, F_norm[:, m], expensive=True)
            self.goodness.append(mean_scores[best_i])
            self.gp.append(best)

    def k_fold_cross_validation(self, items, k, randomize=False):

        if randomize:
            items = list(items)
            shuffle(items)

        slices = [items[i::k] for i in range(k)]

        for i in range(k):
            validation = slices[i]
            training = [item
                        for s in slices if s is not validation
                        for item in s]
            yield list(training), list(validation)

    def predict(self, X):
        n = len(X)
        f = np.zeros((n, len(self.gp)))
        std = np.zeros((n, len(self.gp)))
        for j in range(len(self.gp)):

            X = (X - self.x_min) / (self.x_max - self.x_min)

            mean, _std = self._predict(self.gp[j], X)
            f[:, j] = mean
            f[:,:] = f * (self.f_max - self.f_min) + self.f_min

            std[:, j] = _std
            #std[:, :] = std * (self.f_max - self.f_min)

        return f, std
