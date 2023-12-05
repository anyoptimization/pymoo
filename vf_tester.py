
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize

from pymoo.util import value_functions as vf
import numpy as np

P = np.array([[3.6, 3.9], 
              [2.5, 4.1],    
              [5.5, 2.5],      
              [0.5, 5.2],     
              [6.9, 1.8]])

ranks = [1,2,3,4,5]


linear_vf = vf.linear_vf

vf_prob = vf.OptimizeVF(P, ranks, linear_vf)

#algorithm = GA(pop_size=100)
algorithm = PatternSearch(pop_size=100)

res = minimize(vf_prob,
           algorithm,
           ('n_gen', 200),
               seed=1)

x = np.reshape(res.X[0:-1], (1, -1))

vf.plot_linear_vf(P, x)

print(res.X)



