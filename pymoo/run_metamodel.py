import os

import numpy as np
from pandas import DataFrame

from configuration import Configuration
from metamodels.gpflow_metamodel import GPFlowMetamodel
from metamodels.gpy_metamodel import GPyMetaModel
from metamodels.sklearn_metamodel_dace import SKLearnDACEMetaModel
from operators.lhs_factory import LHS
from operators.random_factory import RandomFactory
from problems.zdt import ZDT1, ZDT2, ZDT3
from util.misc import load_files


def create_data():
    folder = os.path.join(Configuration.BENCHMARK_DIR, "metamodels")

    for prob in [ZDT1(n_var=5), ZDT2(n_var=5), ZDT3(n_var=5)]:

        for i in range(10):

            prefix = os.path.join(folder, prob.__class__.__name__) + "_%s" % (i+1)

            X_test = RandomFactory().sample(500, prob.xl, prob.xu)
            np.savetxt(prefix + ".x_test", X_test)

            F_test, _ = prob.evaluate(X_test)
            np.savetxt(prefix + ".f_test", F_test[:, 1])

            X_train = LHS().sample(50, prob.xl, prob.xu)
            np.savetxt(prefix + ".x_train", X_train)

            F_train, _ = prob.evaluate(X_train)
            np.savetxt(prefix + ".f_train", F_train[:, 1])


if __name__ == '__main__':

    #create_data()
    #exit()

    files = load_files(os.path.join(Configuration.BENCHMARK_DIR, "metamodels"), ".*.x_test", columns=["set"])

    for entry in files:

        X_test = np.loadtxt(entry["path"])

        prefix = entry["path"].split(".")[0]
        X_train = np.loadtxt(prefix + ".x_train")
        F_train = np.array([np.loadtxt(prefix + ".f_train")]).T


        def get_params():
            return [
                ['SKLearnDACEMetaModel', SKLearnDACEMetaModel()],
                # ['SKLearnMetaModel', SKLearnMetaModel()],
                ['GPyMetaModel-one', GPyMetaModel()],
                ['GPFlowMetamodel-linear', GPFlowMetamodel()]
            ]


        for i in range(len(get_params())):
            param = get_params()[i]

            name = param[0]
            metamodel = param[1]
            metamodel.fit(X_train, F_train)
            F_test, _ = metamodel.predict(X_test)

            fname = os.path.join(Configuration.BENCHMARK_DIR, "metamodels",
                                 "%s_%s.out" % (name, entry['fname'].split('.')[0]))
            np.savetxt(fname, F_test)
            print(fname)
