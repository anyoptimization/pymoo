import os
import sys
import time

import numpy as np
from pymop.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from pymop.dtlz import DTLZ2

from pymoo.algorithms.ERNSGAII import RNSGAIII
from pymoo.model.evaluator import Evaluator
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover

from pymoo.indicators.rmetric import RMetric

n_runs = 30
output = os.path.join('.', 'benchmark/')
if not os.path.exists(output):
    os.makedirs(os.getcwd()+'/benchmark/')

def ZDT_Test():
    ref_dirs = {ZDT1: [[(0.2, 0.4), (0.8, 0.4)],
                         [(0.2, 0.6), (0.4, 0.6), (0.5, 0.2), (0.7, 0.2), (0.9, 0)]],
                ZDT2: [[(0.2, 0.8), (0.7, 1), (0.8, 0.2)]],
                ZDT3: [[(0.1, 0.6), (0.3, 0.2), (0.7, -0.25)]]}
    crossover = SimulatedBinaryCrossover(10)

    p = []
    name_list = ["ZDT1_1", "ZDT1_2", "ZDT2_1", "ZDT3_1"]
    for problem, ref_points in ref_dirs.items():
        for i, points in enumerate(ref_points):

            sublist = []
            for algorithm in [RNSGAIII("real", pop_size=100, ep=0.001, crossover=crossover, ref_dirs=points, verbose=0)]:
                for run in range(1, n_runs + 1):
                    name = problem.__class__.__name__ + '_' + str(i) + '_' + str(run)
                    sublist.append((algorithm, problem, run, points, name))
                    # yield (algorithm, problem, run)
            p.append(sublist)
    return p, name_list

def DTLZ_test():
    ref_dirs = {DTLZ2: [{"n_var": 11, "n_obj": 3, "ref_points": [[[0.2, 0.2, 0.6], [0.8, 0.8, 0.8]]]},
                          {"n_var": 14, "n_obj": 5, "ref_points": [[[0.5, 0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.2, 0.2, 0.8]]]},
                          {"n_var": 19, "n_obj": 10, "ref_points": [[[0.25 for i in range(10)]]]}
                          ]
                }
    crossover = SimulatedBinaryCrossover(10)
    p = []
    name_list = ["DTLZ2_1", "DTLZ2_2", "DTLZ2_3"]
    for problem, setup in ref_dirs.items():
        for params in setup:
            parameters = {"n_var": params["n_var"], "n_obj": params["n_obj"]}
            prob = problem(**parameters)
            for i, points in enumerate(params["ref_points"]):
                sublist = []
                for algorithm in [RNSGAIII("real", pop_size=100, ep=0.01, crossover=crossover, ref_dirs=points, verbose=0)]:
                    for run in range(1, n_runs + 1):
                        name = problem.__class__.__name__ + '_' + str(i) + '_' + str(run)
                        sublist.append((algorithm, prob, run, points, name))
                        # yield (algorithm, prob, run)
            p.append(sublist)
    return p, name_list

if __name__ == '__main__':
    param1, name1 = ZDT_Test()
    param2, name2 = DTLZ_test()

    params = param1 + param2
    names = name1 + name2
    n_gen = 500

    results = []

    id = int(sys.argv[1])
    print("Run for parameter settings: %s" % (id))

    run = params[id]
    name = names[id]
    output = output + name
    if not os.path.exists(output):
        os.makedirs(output)

        print(output)

    i = 0
    for algorithm, problem, run, points, n in run:
        try:
            problem = problem()
        except TypeError:
            pass
        points = np.array(points)
        eval = Evaluator(100 * n_gen)
        start_time = time.time()
        X, F, G = algorithm.solve(problem, evaluator=eval, seed=run)
        print("--- %s seconds ---" % (time.time() - start_time))
        fname = os.path.join(output, problem.name()+"_"+str(i))
        print(fname)
        np.savetxt(fname + ".out", F)
        if id <= 5:
            metric_time = time.time()
            igd, hv = RMetric(curr_pop=F, whole_pop=F, ref_points=points, problem=problem).calc()
            results.append(np.array([run, igd, hv]))
            print(igd, " - ", hv)
            print("--- %s seconds ---" % (time.time() - metric_time))
        break
        i += 1

    results = np.array(results)
    if id <=5:
        np.savetxt(os.path.join(output, "results.out"), results, fmt=("%-3i","%-10.7f","%-10.7f"))
    np.savetxt(os.path.join(output, "points.out"), points, fmt="%-5.2f")