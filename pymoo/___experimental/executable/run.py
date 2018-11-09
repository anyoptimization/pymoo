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
    from pymoo.util.non_dominated_sorting import NonDominatedSorting
    from pymoo.indicators.igd import IGD

    if len(sys.argv) < 2:
        raise Exception("Usage: python pyford.py <dat> [<out>]")

    # load the data for the experiment
    fname = sys.argv[1]
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    problem, algorithm, termination, seed, pf = data['problem'], data['algorithm'], data['termination'], data['seed'], \
                                                data['pf']

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
        I = NonDominatedSorting().do(F, only_non_dominated_front=True)
        igd = IGD(pf, normalize=True).calc(F[I, :])

        elapsed = (time.time() - start_time)
        print(fname, "in --- %s seconds ---" % elapsed)

        # create directory of necessary
        os.makedirs(os.path.dirname(out), exist_ok=True)

        np.savetxt(out, F)

        if len(sys.argv) == 4:
            out_dat = sys.argv[3]

            hist = res['history']

            H = []
            _igd = np.zeros(len(hist))
            nadir_point = np.zeros((len(hist), problem.n_obj))
            ideal_point = np.zeros((len(hist), problem.n_obj))

            for i, e in enumerate(hist):
                nadir_point[i, :] = e.survival.nadir_point
                ideal_point[i, :] = e.survival.ideal_point

                _F = e.D['pop'].F
                I = NonDominatedSorting().do(_F, only_non_dominated_front=True)
                _igd[i] = IGD(pf, normalize=False).calc(_F[I, :])
                H.append(_F[I, :])

            D = {'F': F, 'igd': F, '_igd': _igd, 'nadir_point': nadir_point, 'ideal_point': ideal_point, 'H': H}

            pickle.dump(D, open(out_dat, 'wb'))

    except Exception as e:
        print(e)
        print("Error: %s" % fname)

    # pf = problem.pareto_front()
    # print(IGD(pf).calc(F))
    # print(IGD(pf).calc(_F))
