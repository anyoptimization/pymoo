from metamodels.metamodel import MetaModel
from metamodels.selection_error_probablity import selection_error_probability
from operators.random_factory import RandomFactory
import numpy as np

from operators.random_spare_factory import RandomSpareFactory
from problems.rastrigin import Rastrigin
from problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4

if __name__ == '__main__':
    prob = ZDT1(n_var=5)

    X = RandomSpareFactory().sample(30, prob.xl, prob.xu)
    F = np.array([prob.evaluate(X[i,:])[0] for i in range(len(X))])

    meta_model = MetaModel()
    meta_model.fit(X, F)

    X_test = RandomFactory().sample(400, prob.xl, prob.xu)
    F_test_true = np.array([prob.evaluate(X_test[i,:])[0] for i in range(len(X_test))])
    F_test_pred = meta_model.predict(X_test)

    print np.power(F_test_true - F_test_pred, 2).mean()
    print selection_error_probability(F_test_true, F_test_pred)
