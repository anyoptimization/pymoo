from multiprocessing.pool import ThreadPool
from time import sleep

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.parallelization import StarmapParallelization


class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum()
        sleep(0.0005)

# the number of threads to be used
n_threads = 8

# initialize the pool
pool = ThreadPool(n_threads)

runner = StarmapParallelization(pool.starmap)

# define the problem by passing the starmap interface of the thread pool
problem = MyProblem(elementwise_runner=runner)

res = minimize(problem, PSO(), seed=1, n_gen=100, verbose=True)
print('Threads:', res.exec_time)

pool.close()
