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

    sys.path.append('/home/blankjul/workspace/pymop')
    import pymop

    if len(sys.argv) < 2:
        raise Exception("Usage: python run.py <dat> [<out>]")

    # load the data for the experiment
    fname = sys.argv[1]
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    problem, algorithm, evaluator, seed = data['problem'], data['algorithm'], data['evaluator'], data['seed']

    if len(sys.argv) == 2:
        tmp = sys.argv[1]
        out = tmp[:tmp.find('.')] + '.out'
    else:
        out = sys.argv[2]

    start_time = time.time()
    try:

        X, F, G = algorithm.solve(problem,
                                  evaluator=evaluator,
                                  seed=seed,
                                  save_history=(len(sys.argv) == 4),
                                  return_only_feasible=False,
                                  return_only_non_dominated=False,
                                  disp=False)

        elapsed = (time.time() - start_time)
        print(fname, "in --- %s seconds ---" % elapsed)

        np.savetxt(out, F)

        if len(sys.argv) == 4:
            out_dat = sys.argv[3]
            with open(out_dat, 'wb') as f:
                pickle.dump({'hist': algorithm.history}, f)

    except:
        print("Error: %s" % fname)

    # pf = problem.pareto_front()
    # print(IGD(pf).calc(F))
    # print(IGD(pf).calc(_F))
