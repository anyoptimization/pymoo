import sklearn
from gpflow.mean_functions import Linear
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic, Matern
import gpflow
from metamodels.metamodel import MetaModel
import numpy


class GPFlowMetamodel(MetaModel):

    def _get_parameter(self, d={}):
        n_var = d['n_var']
        return [gpflow.kernels.RBF(input_dim=n_var, ARD=True)]

    def _predict(self, metamodel, X):
        mean, cov = metamodel.predict_y(X)

        mean = mean.T
        cov = cov.T
        if mean.shape[0] == 1:
            mean = mean[0]
        if cov.shape[0] == 1:
            cov = cov[0]
        return mean, cov

    def _create_and_fit(self, parameter, X, F):
        m = gpflow.gpr.GPR(X, numpy.array([F]).T, kern=parameter, mean_function=Linear(numpy.ones((X.shape[1], 1)), numpy.ones((1,1))))
        m.optimize()
        return m
