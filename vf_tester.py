from pymoo.util import value_functions as mvf
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


## Evolutionary approach: 

vf = mvf.create_linear_vf(P, ranks, "ES")

mvf.plot_vf(P, vf)

## Scipy solver methods 
vf = mvf.create_linear_vf(P, ranks, "scimin")

mvf.plot_vf(P, vf)



