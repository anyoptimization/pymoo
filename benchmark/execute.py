import os
import pickle
import sys
import time
import traceback

import numpy as np

if __name__ == '__main__':

    # insert this project in path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    import sys
    sys.path.insert(0, "/home/blankjul/workspace/pymoo/")
    sys.path.insert(0, "/Users/julesy/workspace/pymoo/")

    from pymoo.optimize import minimize

    # load the data for the benchmark
    fname = sys.argv[1]

    with open(fname, 'rb') as f:
        data = pickle.load(f)

    start_time = time.time()
    try:

        if 'algorithm' in data:
            res = data['algorithm'].solve(data['problem'], data['termination'], seed=data['seed'])
            problem = data['problem']

        else:
            res = minimize(
                *data['args'],
                **data['kwargs']
            )
            problem = data['args'][0]

        elapsed = (time.time() - start_time)
        print(fname, "in --- %s seconds ---" % elapsed)

        # create directory if necessary
        out = sys.argv[2]
        os.makedirs(os.path.dirname(out), exist_ok=True)

        # if a feasible solution has been found
        """
        if res.F is not None:
            F = res.F
            M = F
            if problem.n_constr > 0:
                M = np.hstack([F, res.CV])

        # if no feasible solution was found
        else:
            F, CV = res.pop.get("F", "CV")
            best = np.argmin(CV)
            M = np.array([F[best], CV[best]])
        """

        # M = np.column_stack([res.pop.get("F"), res.pop.get("CV"), res.pop.get("X")])
        M = res.pop.get("F")
        np.savetxt(out, M)

        # np.savetxt(out + ".gen", np.array([res.algorithm.n_gen]))


    except Exception as e:
        traceback.print_exc()
        print(e)
        print("Error: %s" % fname)
