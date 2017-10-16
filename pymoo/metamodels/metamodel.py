from random import shuffle

import numpy as np

from util.misc import calc_mse


class MetaModel:
    def __init__(self):
        self.parameters = []
        self.gp = []
        self.goodness = []

    def _get_parameter(self, d={}):
        pass

    def _predict(self, metamodel, X):
        pass

    def _create_and_fit(self, parameter, X, F):
        pass

    def fit(self, X, F):

        # only unique rows for X
        y = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
        _, idx = np.unique(y, return_index=True)
        X = X[idx]
        F = F[idx]

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
                    for train, test in self.k_fold_cross_validation(range(n_observations), 5, False):
                        # fit the meta model and predict results
                        model = self._create_and_fit(par, X[train, :], F[train, m])
                        f_hat, _ = self._predict(model, X[test, :])
                        f_true = F[test, m]

                        # calc metric for this run
                        metric = calc_mse(f_true, f_hat)
                        scores.append(metric)

                    # save the goodness for this metamodel
                    mean_scores.append(np.mean(scores))

            # get the best metamodel for this objective
            best_i = np.argmin(mean_scores)

            # train it on all training data
            par = self._get_parameter(d)[best_i]
            self.parameters.append(par)

            best = self._create_and_fit(par, X, F[:, m])
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
            mean, _std = self._predict(self.gp[j], X)
            f[:, j] = mean
            std[:, j] = _std

        return f, std
