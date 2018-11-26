import os
import pickle
import sys
import time

import numpy as np



if __name__ == '__main__':

    # insert this project in path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    import sys

    sys.path.insert(0, "/home/blankjul/workspace/pymop/")
    sys.path.insert(0, "/Users/julesy/workspace/pymop/")
    sys.path.insert(0, "/home/blankjul/workspace/pymoo/")
    sys.path.insert(0, "/Users/julesy/workspace/pymoo/")
    import pymop
    from pymoo.optimize import minimize

    # load the data for the experiments
    fname = sys.argv[1]
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    start_time = time.time()
    try:

        if 'algorithm' in data:
            res = data['algorithm'].solve(data['problem'], data['termination'], seed=data['seed'])

        else:
            res = minimize(
                *data['args'],
                **data['kwargs']
            )

        F = res.F

        elapsed = (time.time() - start_time)
        print(fname, "in --- %s seconds ---" % elapsed)



        # create directory if necessary
        out = sys.argv[2]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savetxt(out, F)

    except Exception as e:
        print(e)
        print("Error: %s" % fname)
