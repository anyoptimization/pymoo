import numpy as np



## General I/O: 

# Input: Most abstract: user does pair wise-comparison of points

#       In between: if no pair can be see as better than others, either the set of solutions is t


# Input less abstract 1: A list of non-dominated points 
# Input less abstract 2: The ranking of the given non-dominated points 


def create_value_fnc(F, rankings): 

    return lambda f_new:  np.sum(f_new)


def test_sum(a, b):
    return a + b 









