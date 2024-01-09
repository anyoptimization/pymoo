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

# This is a great example. Illustrates the when polynomial versus linear works 
# when the problem is changed from. 
# TODO Also doesn't work for scimin minimization. It does that weird thing where
# multiple lines are drawn on the contour
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

# linear or poly
fnc_type = "poly"

# max (False) or min (True)
minimize = False


if fnc_type == "linear":

    vf_res = mvf.create_linear_vf(P, ranks, approach, minimize)

elif fnc_type == "poly": 

    vf_res = mvf.create_poly_vf(P, ranks, approach, minimize)

else: 

    print("function not supported")

print("Final parameters:")
print(vf_res.params)

print("Final epsilon:")
print(vf_res.epsilon)

mvf.plot_vf(P, vf_res.vf)




