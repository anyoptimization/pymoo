import os
import pickle
import sys
import time

import numpy as np

from pymoo.optimize import minimize

if __name__ == '__main__':

    # insert this project in path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    import sys

    sys.path.append("/home/blankjul/workspace/pymop/")
    sys.path.append("/Users/julesy/workspace/pymop/")
    sys.path.append("/home/blankjul/workspace/pymoo/")
    sys.path.append("/Users/julesy/workspace/pymoo/")
    import pymop

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

        out = data['out']
        if len(sys.argv) == 3:
            out = os.path.join(sys.argv[2], out)

        # create directory of necessary
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savetxt(out, F)

    except Exception as e:
        print(e)
        print("Error: %s" % fname)
