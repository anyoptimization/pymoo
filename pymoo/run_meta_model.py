import numpy as np

from problems.zdt import ZDT1
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

from problems.rastrigin import Rastrigin

if __name__ == '__main__':

    sp = samplingplan(2)
    X = sp.rlh(30)

    prob = ZDT1(2)

    Y = np.array([prob.evaluate(X[i, :])[0] for i in range(len(X))])

    print Y
    k = kriging(X, Y)
    k.train()



    X_test = np.random.random([100, 2])
    X_true = prob.evaluate(X_test)

    print k.predict(np.array([.25, .25]))

    # And plot the results
    k.plot()
