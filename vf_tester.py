from pymoo.util import value_functions as mvf
import numpy as np

## Examples from Sinha's code
#P = np.array([[3.6, 3.9], 
#              [2.5, 4.1],    
#              [5.5, 2.5],      
#              [0.5, 5.2],     
#              [6.9, 1.8]])
#ranks = [1,2,3,4,5]
#ranks = [1,4,3,5,2]

## Domination example values 
P = np.array([[1, 5],
              [2, 3],
              [3, 2],
              [4, 1]])

ranks = [3,4,2,1]


#P = np.array([[4, 4], 
#              [4, 3], 
#              [2, 4], 
#              [4, 1], 
#              [1, 3]]);
#
#ranks = [1,2,3,4,5]


# ES or scimin
approach = "ES"

# Linear or polynomial 
fnc_type = "linear"


if fnc_type == "linear":

    vf_res = mvf.create_linear_vf(P, ranks, approach)

elif fnc_type == "poly": 

    vf_res = mvf.create_poly_vf(P, ranks, approach)

else: 

    print("function not supported")

print("Final parameters:")
print(vf_res.params)

print("Final epsilon:")
print(vf_res.epsilon)

mvf.plot_vf(P, vf_res.vf)




