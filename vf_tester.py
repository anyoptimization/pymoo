from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize as scimin
from pymoo.optimize import minimize

from pymoo.util import value_functions as vf
import numpy as np

#P = np.array([[3.6, 3.9], 
#              [2.5, 4.1],    
#              [5.5, 2.5],      
#              [0.5, 5.2],     
#              [6.9, 1.8]])
#
#ranks = [1,2,3,4,5]
#ranks = [1,4,3,5,2]


P = np.array([[4, 4], 
              [4, 3], 
              [2, 4], 
              [4, 1], 
              [1, 3]]);

ranks = [1,2,3,4,5]



linear_vf = vf.linear_vf

vf_prob = vf.OptimizeVF(P, ranks, linear_vf)


## Evolutionary approach: 

#algorithm = GA(pop_size=100)
#algorithm = PatternSearch(pop_size=100)
#algorithm = DE(pop_size=100)
algorithm = ES()

res = minimize(vf_prob,
           algorithm,
           ('n_gen', 200),
               seed=1)

print("Variables: %s" % res.X[0:-1])
print("Epsilon: %s" % res.X[-1])

x = np.reshape(res.X[0:-1], (1, -1))

vf.plot_linear_vf(P, x)



## Scipy solver methods 
# Inequality constraints
lb = [-np.inf] * (P.shape[0] - 1)
ub = [0] * (P.shape[0] - 1)

# Equality constraints
lb.append(0)
ub.append(0)

constr = NonlinearConstraint(vf_prob.build_constr(), lb, ub)

x0 = [0.5, 0.5, 0.5]

res = scimin(vf_prob._obj_func, x0, constraints= constr)

print("Variables: %s" % res.x[0:-1])
print("Epsilon: %s" % res.x[-1])

x = np.reshape(res.x[0:-1], (1, -1))

vf.plot_linear_vf(P, x)



