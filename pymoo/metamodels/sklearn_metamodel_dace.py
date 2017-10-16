
import warnings
import numpy as np
from sklearn.gaussian_process.gaussian_process import GaussianProcess

from metamodels.metamodel import MetaModel


class SKLearnDACEMetaModel(MetaModel):

    def _gen_parameter(self):
        for corr in ['absolute_exponential', 'squared_exponential', 'cubic', 'linear']:
            for regr in ['constant', 'linear', 'quadratic']:
                yield [regr, corr]

    def _get_parameter(self, d={}):
        return list(self._gen_parameter())

    def _predict(self, metamodel, X):
        return metamodel.predict(X, eval_MSE=True)

    def _create_and_fit(self, parameter, X, F):
        warnings.filterwarnings('ignore')
        gp = GaussianProcess(regr=parameter[0], corr=parameter[1],random_start=5,nugget=50. * np.finfo(np.double).eps,
                                                      theta0=10, thetaL=0.0001, thetaU=20)
        return gp.fit(X, F)
