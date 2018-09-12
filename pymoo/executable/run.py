import os
import pickle
import sys
import time

import numpy as np

if __name__ == '__main__':
    """
    This class performs one run given a pickled object saving all the necessary data.
    """

    # insert this project in path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    import sys

    sys.path.append("/home/blankjul/workspace/pymop/")
    sys.path.append("/Users/julesy/workspace/pymop/")

    import pymop
    from pymop.problems.dtlz import DTLZ1

    if len(sys.argv) < 2:
        raise Exception("Usage: python run.py <dat> [<out>]")

    # load the data for the experiment
    fname = sys.argv[1]
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    problem, algorithm, termination, seed = data['problem'], data['algorithm'], data['termination'], data['seed']

    if len(sys.argv) == 2:
        tmp = sys.argv[1]
        out = tmp[:tmp.find('.')] + '.out'
    else:
        out = sys.argv[2]

    start_time = time.time()
    try:

        res = algorithm.solve(problem,
                                  termination=termination,
                                  seed=seed,
                                  save_history=(len(sys.argv) == 4),
                                  return_only_feasible=False,
                                  return_only_non_dominated=False,
                                  disp=False)

        F = res['F']

        elapsed = (time.time() - start_time)
        print(fname, "in --- %s seconds ---" % elapsed)

        # create directory of necessary
        os.makedirs(os.path.dirname(out), exist_ok=True)

        np.savetxt(out, F)

        if len(sys.argv) == 4:
            out_dat = sys.argv[3]
            with open(out_dat, 'wb') as f:
                pickle.dump({'hist': algorithm.history}, f)

    except Exception as e:
        print(e)
        print("Error: %s" % fname)

    # pf = problem.pareto_front()
    # print(IGD(pf).calc(F))
    # print(IGD(pf).calc(_F))
