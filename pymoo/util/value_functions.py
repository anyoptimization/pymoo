import numpy as np

from pymoo.core.problem import Problem
import pymoo.algorithms.soo.nonconvex.ga

# Notes: 
# I'm using the suffix _vf to denote variables used in the value-function optimization 
# This is to avoid confuse it with variables being optimized in the main function 

# Input 1: A list of non-dominated points 
# Input 2: The ranking of the given non-dominated points 
# Input 3: constraint function for optimizing the value func
# Input 4: the skeleton utility function that we're trying to optimize
def create_vf(F, ranks, cFunc, utilFunc): 

    # Combine rankings and objective functions into one matrix 
    # The first O columns are the objective values, and the O + 1 column is the ranking

    X = np.hstack((F, np.array([ranks]).T))


    return lambda f_new:  np.sum(f_new)



def create_linear_vf(F, ranks): 

    return lambda f_new:  np.sum(f_new)


def linear_vf(F, x_vf): 

    return np.multiply(F, x_vf)


#def linear_const(F, x_vf):

class OptimizeVF(Problem): 


    def __init__(self, F, ranks):
       
        # One var for each dimension of the object space, plus epsilon 
        n_var_vf = np.size(F, 0) + 1

        # it has one inequality constraints per dimension in F, and one equality 
        n_ieq_c_vf = n_var_vf - 1
       
        xl_vf = [0.0] * n_var_vf 
        xu_vf = [1.0] * n_var_vf
        
        # upper/lower bound on the epsilon variable is -1000/1000
        xl_vf[-1] = -1000
        xu_vf[-1] = 1000

        # TODO start everything at 0.5

        # Add the rankings onto our objectives 
        self.P = np.hstack((F, np.array([ranks]).T))

        # Sort P by rankings in the last column
        self.P = self.P[self.P[:, -1].argsort()]

        super().__init__(n_var_vf, n_obj=1, n_ieq_constr=n_ieq_c_vf, n_eq_constr=1, xl=xl_vf, xu=xu_vf)

       

    def _evaluate(self, x, out, *args, **kwargs):

        ## Objective function: 

        # maximize epsilon, or the minimum distance between each contour 
        out["F"] = -x[:,-1]

        ## Inequality
        # TODO for now, assuming there are no ties in the ranks
        out["G"] = -99
        
            
        ## Equality
        
        out["H"] = -99




