import pickle

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.evaluator import Evaluator
from pymop.problems.zdt import ZDT4

if __name__ == '__main__':
    n_gen = 200
    pop_size = 100

    problem = ZDT4()
    algorithm = NSGA2(pop_size=pop_size)
    evaluator = Evaluator(pop_size * n_gen)
    seed = 0

    fname = 'pynsga2_zdt4_0.dat'

    data = {'problem': problem, 'algorithm': algorithm, 'evaluator': evaluator, 'seed': seed}
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
