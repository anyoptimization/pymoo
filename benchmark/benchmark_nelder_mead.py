"""
This is the experiment for nsga2.
"""
import os
import pickle

from pymoo.algorithms.so_nelder_mead import NelderMead

from pymoo.factory import get_problem

setup = [
    "go-amgm",
    "go-ackley01",
    "go-ackley02",
    "go-ackley03",
    "go-adjiman",
    "go-alpine01",
    "go-alpine02",
    "go-bartelsconn",
    "go-beale",
    "go-biggsexp02",
    "go-biggsexp03",
    "go-biggsexp04",
    "go-biggsexp05",
    "go-bird",
    "go-bohachevsky1",
    "go-bohachevsky2",
    "go-bohachevsky3",
    "go-boxbetts",
    "go-branin01",
    "go-branin02",
    "go-brent",
    "go-brown",
    "go-bukin02",
    "go-bukin04",
    "go-bukin06",
    "go-carromtable",
    "go-chichinadze",
    "go-cigar",
    "go-cola",
    "go-colville",
    "go-corana",
    "go-cosinemixture",
    "go-crossintray",
    "go-crosslegtable",
    "go-crownedcross",
    "go-csendes",
    "go-cube",
    "go-damavandi",
    "go-devilliersglasser01",
    "go-devilliersglasser02",
    "go-deb01",
    "go-deb03",
    "go-decanomial",
    "go-deceptive",
    "go-deckkersaarts",
    "go-deflectedcorrugatedspring",
    "go-dixonprice",
    "go-dolan",
    "go-dropwave",
    "go-easom",
    "go-eckerle4",
    "go-eggcrate",
    "go-eggholder",
    "go-elattarvidyasagardutta",
    "go-exp2",
    "go-exponential",
    "go-freudensteinroth",
    "go-gear",
    "go-giunta",
    "go-goldsteinprice",
    "go-griewank",
    "go-gulf",
    "go-hansen",
    "go-hartmann3",
    "go-hartmann6",
    "go-helicalvalley",
    "go-himmelblau",
    "go-holdertable",
    "go-hosaki",
    "go-infinity",
    "go-jennrichsampson",
    "go-judge",
    "go-katsuura",
    "go-keane",
    "go-kowalik",
    "go-langermann",
    "go-lennardjones",
    "go-leon",
    "go-levy03",
    "go-levy05",
    "go-levy13",
    "go-matyas",
    "go-mccormick",
    "go-meyer",
    "go-michalewicz",
    "go-mielecantrell",
    "go-mishra01",
    "go-mishra02",
    "go-mishra03",
    "go-mishra04",
    "go-mishra05",
    "go-mishra06",
    "go-mishra07",
    "go-mishra08",
    "go-mishra09",
    "go-mishra10",
    "go-mishra11",
    "go-multimodal",
    "go-needleeye",
    "go-newfunction01",
    "go-newfunction02",
    "go-oddsquare",
    "go-parsopoulos",
    "go-pathological",
    "go-paviani",
    "go-penholder",
    "go-penalty01",
    "go-penalty02",
    "go-permfunction01",
    "go-permfunction02",
    "go-pinter",
    "go-plateau",
    "go-powell",
    "go-powersum",
    "go-price01",
    "go-price02",
    "go-price03",
    "go-price04",
    "go-qing",
    "go-quadratic",
    "go-quintic",
    "go-rana",
    "go-rastrigin",
    "go-ratkowsky01",
    "go-ratkowsky02",
    "go-ripple01",
    "go-ripple25",
    "go-rosenbrock",
    "go-rosenbrockmodified",
    "go-rotatedellipse01",
    "go-rotatedellipse02",
    "go-salomon",
    "go-sargan",
    "go-schaffer01",
    "go-schaffer02",
    "go-schaffer03",
    "go-schaffer04",
    "go-schwefel01",
    "go-schwefel02",
    "go-schwefel04",
    "go-schwefel06",
    "go-schwefel20",
    "go-schwefel21",
    "go-schwefel22",
    "go-schwefel26",
    "go-schwefel36",
    "go-shekel05",
    "go-shekel07",
    "go-shekel10",
    "go-shubert01",
    "go-shubert03",
    "go-shubert04",
    "go-sineenvelope",
    "go-sixhumpcamel",
    "go-sodp",
    "go-sphere",
    "go-step",
    "go-step2",
    "go-stochastic",
    "go-stretchedv",
    "go-styblinskitang",
    "go-testtubeholder",
    "go-threehumpcamel",
    "go-thurber",
    "go-treccani",
    "go-trefethen",
    "go-trid",
    "go-trigonometric01",
    "go-trigonometric02",
    "go-tripod",
    "go-ursem01",
    "go-ursem03",
    "go-ursem04",
    "go-ursemwaves",
    "go-ventersobiezcczanskisobieski",
    "go-vincent",
    "go-watson",
    "go-wavy",
    "go-wayburnseader01",
    "go-wayburnseader02",
    "go-weierstrass",
    "go-whitley",
    "go-wolfe",
    "go-xinsheyang01",
    "go-xinsheyang02",
    "go-xinsheyang03",
    "go-xinsheyang04",
    "go-xor",
    "go-yaoliu04",
    "go-yaoliu09",
    "go-zacharov",
    "go-zerosum",
    "go-zettl",
    "go-zimmerman",
    "go-zirilli"
]


def add_with_variables(D, problem, n_vars):
    for n_var in n_vars:
        D[problem + "-%02d" % n_var] = get_problem(problem, n_var=n_var)


if __name__ == '__main__':

    # all the files to be run in a list
    run_files = []

    # prefix of the folder to save the files
    prefix = "runs"

    # name of the experiment
    name = "nelder-mead-0.3.2"

    # number of runs to execute
    n_runs = 10

    # path were the files for this experiment are saved
    path = os.path.join(prefix, name)

    for _problem in setup:

        problem = get_problem(_problem)

        method = NelderMead(n_max_local_restarts=2)

        for run in range(1, n_runs + 1):
            fname = "%s_%s.run" % (_problem, run)
            _in = os.path.join(path, fname)
            _out = "results/%s/%s/%s_%s.out" % (name, _problem.replace("_", "/"), _problem, run)

            data = {
                'args': [problem, method],
                'kwargs': {
                    'seed': run,
                },
                'in': _in,
                'out': _out,
            }

            os.makedirs(os.path.join(os.path.dirname(_in)), exist_ok=True)

            with open(_in, 'wb') as f:
                pickle.dump(data, f)
                run_files.append(data)

        # create the final run.txt file
        with open(os.path.join(prefix, name, "run.bat"), 'w') as f:
            for run_file in run_files:
                f.write("python execute.py %s %s\n" % (run_file['in'], run_file['out']))
