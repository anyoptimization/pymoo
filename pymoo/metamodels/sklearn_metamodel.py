import sklearn
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic, Matern

from metamodels.metamodel import MetaModel


class SKLearnMetaModel(MetaModel):

    def _get_parameter(self, d={}):
        return [RationalQuadratic(), RBF(), ConstantKernel(), Matern()]

    def _predict(self, metamodel, X):
        return metamodel.predict(X, return_std=True)

    def _create_and_fit(self, parameter, X, F):
        metamodel = sklearn.gaussian_process.GaussianProcessRegressor(kernel=parameter, normalize_y=True,
                                                                      n_restarts_optimizer=5)
        metamodel.fit(X, F)
        return metamodel
