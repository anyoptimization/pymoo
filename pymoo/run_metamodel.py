import argparse
import os

import numpy as np
import sys

from configuration import Configuration
from metamodels.gpflow_metamodel import GPFlowMetamodel
from metamodels.gpy_metamodel import GPyMetaModel
from metamodels.sklearn_metamodel_dace import SKLearnDACEMetaModel
from metamodels.tensorflow_metamodel import TensorFlowMetamodel
from operators.lhs_factory import LHS
from operators.random_factory import RandomFactory
from problems.ackley import Ackley
from problems.rastrigin import Rastrigin
from problems.zdt import ZDT1, ZDT2, ZDT3
from util.misc import load_files


def create_data():
    folder = os.path.join(Configuration.BENCHMARK_DIR, "metamodels")

    Rastrigin().evaluate(np.zeros((1, 2)))
    for prob in [Rastrigin(), Ackley()]:
    #for prob in [ZDT1(n_var=5), ZDT2(n_var=5), ZDT3(n_var=5), Rastrigin(), Ackley()]:

        for i in range(10):

            prefix = os.path.join(folder, prob.__class__.__name__) + "_%s" % (i+1)

            X_test = RandomFactory().sample(500, prob.xl, prob.xu)
            np.savetxt(prefix + ".x_test", X_test)

            F_test, _ = prob.evaluate(X_test)
            np.savetxt(prefix + ".f_test", F_test[:, 0])

            X_train = LHS().sample(50, prob.xl, prob.xu)
            np.savetxt(prefix + ".x_train", X_train)

            F_train, _ = prob.evaluate(X_train)
            np.savetxt(prefix + ".f_train", F_train[:, 0])


if __name__ == '__main__':

    #create_data()
    #exit()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n', metavar='n', default=None, type=str, help='Execute a specific run (HPCC)')
    parser.add_argument('-o', '--out', default=None, help='Defines the storage of the output file(s).')
    args = parser.parse_args()

    n = sys.argv[1]
    counter = 0

    #files = load_files(os.path.join(Configuration.BENCHMARK_DIR, "metamodels"), "ford.*.x_test", columns=["set", "run"])
    files = load_files(args.out, ".*.x_test", columns=["set", "run"])

    for entry in files:

        X_test = np.loadtxt(entry["path"])

        prefix = entry["path"].split(".")[0]
        X_train = np.loadtxt(prefix + ".x_train")
        F_train = np.array([np.loadtxt(prefix + ".f_train")]).T


        def get_params():
            return [
                #['DACEMetaModel', SKLearnDACEMetaModel()],
                #['SKLearnMetaModel', SKLearnMetaModel()],
                #['GPyMetaModel-ARD', GPyMetaModel()],
                ['GPFlowMetamodel-Restart', GPFlowMetamodel()],
                #['TensorFlow', TensorFlowMetamodel()]
            ]


        for i in range(len(get_params())):

            counter += 1
            if args.n is not None and counter != int(args.n):
                continue

            param = get_params()[i]

            print("------------------------------")
            print(entry["path"])
            print(param)
            print(entry["run"])
            name = param[0]
            metamodel = param[1]
            metamodel.fit(X_train, F_train)
            F_test, _ = metamodel.predict(X_test)

            #fname = os.path.join(Configuration.BENCHMARK_DIR, "metamodels", "%s_%s.out" % (name, entry['fname'].split('.')[0]))
            fname = os.path.join(args.out, "%s_%s.out" % (name, entry['fname'].split('.')[0]))
            np.savetxt(fname, F_test)
            print(fname)
