from pymoo.util import value_functions as mvf
import numpy as np

## Examples from Sinha's code
P = np.array([[3.6, 3.9], 
              [2.5, 4.1],    
              [5.5, 2.5],      
              [0.5, 5.2],     
              [6.9, 1.8]])
ranks = [1,2,2,3,4]
#ranks = [1,4,3,5,2]

# This is a great example. Illustrates the when polynomial versus linear works 
# when the problem is changed from. 
# TODO Also doesn't work for scimin minimization. It does that weird thing where
# multiple lines are drawn on the contour
#P = np.array([[1, 5],
#              [2, 3],
#              [3, 2],
#              [4, 1]])
#
#ranks = [3,4,2,1]


#P = np.array([[4, 4], 
#              [4, 3], 
#              [2, 4], 
#              [4, 1], 
#              [1, 3]]);
#
#ranks = [1,2,3,4,5]


# Partial ordering example
P = np.array([[3.6, 3.9], [2.5, 4.1], [5.5, 2.5], [0.5, 5.2], [6.9, 1.8]])

ranks = [1, 2, 3, 3, 4]

# opt_method can be trust-constr, SLSQP, ES, or GA
opt_method = "trust-constr"

# linear or poly
fnc_type = "poly"


if fnc_type == "linear":

    vf_res = mvf.create_linear_vf(P, ranks, method=opt_method)

elif fnc_type == "poly": 

    vf_res = mvf.create_poly_vf(P, ranks, method=opt_method)

else: 

    print("function not supported")

if vf_res.fit:

    print("Final parameters:")
    print(vf_res.params)

    print("Final epsilon:")
    print(vf_res.epsilon)

    mvf.plot_vf(P, vf_res.vf)




