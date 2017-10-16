import GPy
import numpy as np

from metamodels.metamodel import MetaModel


class GPyMetaModel(MetaModel):
    def __init__(self):
        super().__init__()


    def _get_parameter(self, d):

        n_var = d['n_var']
        return [GPy.kern.Exponential(n_var) + GPy.kern.RBF(n_var) + GPy.kern.Matern32(n_var) + GPy.kern.Matern52(n_var)+
                 GPy.kern.Linear(n_var)]


    def _predict(self, metamodel, X):
        mean, cov = metamodel.predict_noiseless(X)
        mean = mean.T
        cov = cov.T
        if mean.shape[0] == 1:
            mean = mean[0]
        if cov.shape[0] == 1:
            cov = cov[0]
        return mean, cov

    def _create_and_fit(self, kernel, X, F):
        model = GPy.models.GPRegression(X, np.array([F]).T, kernel=kernel)
        model.optimize()
        return model
