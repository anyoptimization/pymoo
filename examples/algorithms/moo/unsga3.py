import matplotlib.pyplot as plt
import numpy as np

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.optimize import minimize
from pymoo.problems import get_problem

problem = get_problem("ackley", n_var=30)

ref_dirs = np.array([[1.0]])

algorithm = UNSGA3(ref_dirs, pop_size=100)

res = minimize(problem,
               algorithm,
               termination=('n_gen', 150),
               save_history=True,
               seed=1)

print("UNSGA3: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

_res = minimize(problem,
                NSGA3(ref_dirs, pop_size=100),
                termination=('n_gen', 150),
                save_history=True,
                seed=1)
print("NSGA3: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

ret = [algorithm.opt[0].f for algorithm in res.history]
_ret = [algorithm.opt[0].f for algorithm in _res.history]

plt.plot(np.arange(len(ret)), ret, label="unsga3")
plt.plot(np.arange(len(_ret)), _ret, label="nsga3")
plt.title("Convergence")
plt.xlabel("Generation")
plt.ylabel("F")
plt.legend()
plt.show()
