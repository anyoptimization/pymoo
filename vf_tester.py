from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize

from pymoo.util import value_functions as vf
import numpy as np

P = np.array([[3.6, 3.9], 
              [2.5, 4.1],    
              [5.5, 2.5],      
              [0.5, 5.2],     
              [6.9, 1.8]])

#ranks = [1,2,3,4,5]
ranks = [1,4,3,5,2]


#P = np.array([[4, 4], 
#              [4, 3], 
#              [2, 4], 
#              [4, 1], 
#              [1, 3]]);
#
#ranks = [1,2,3,4,5]



linear_vf = vf.linear_vf

vf_prob = vf.OptimizeVF(P, ranks, linear_vf)

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




