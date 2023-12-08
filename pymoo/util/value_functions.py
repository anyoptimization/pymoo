import numpy as np
from pymoo.optimize import minimize as moomin
from scipy.optimize import minimize as scimin
from pymoo.core.problem import Problem
import pymoo.algorithms.soo.nonconvex.ga
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
from pymoo.algorithms.soo.nonconvex.es import ES
# Notes: 
# I'm using the suffix _vf to denote variables used in the value-function optimization 
# This is to avoid confuse it with variables being optimized in the main function 

# Input 1: A list of non-dominated points 
# Input 2: The ranking of the given non-dominated points 
# Input 3: constraint function for optimizing the value func
# Input 4: the skeleton utility function that we're trying to optimize
def create_vf(P, ranks, ineq_constr, vf="linear", algorithm="scimin"): 

    if vf == "linear":
        return create_linear_vf(P, ranks, algorithm)
    else:
        raise ValueError("Value function '%d' not supported." % vf) 
    

    return lambda f_new:  np.sum(f_new)


def create_linear_vf(P, ranks, algorithm="scimin"): 
    
    if algorithm == "scimin": 
        return create_vf_scipy(P, ranks, linear_vf)
    elif algorithm == "ES": 
        return create_vf_pymoo(P, ranks, linear_vf)
    else: 
        raise ValueError("Algorithm %s not supported" % algorithm) 



def create_vf_scipy(P, ranks, vf): 

    # Inequality constraints
    lb = [-np.inf] * (P.shape[0] - 1)
    ub = [0] * (P.shape[0] - 1)

    # Equality constraints
    lb.append(0)
    ub.append(0)

    P_sorted = _sort_P(P, ranks)

    constr = NonlinearConstraint(_build_constr_linear(P_sorted, linear_vf), lb, ub)

    x0 = [0.5, 0.5, 0.5]

    res = scimin(_obj_func, x0, constraints= constr)

    return lambda P_in: vf(P_in, res.x[0:-1])
        

def create_vf_pymoo(P, ranks, vf): 

    vf_prob = OptimizeVF(P, ranks, linear_vf)

    algorithm = ES()

    res = moomin(vf_prob,
        algorithm,
        ('n_gen', 200),
        seed=1)

    return lambda P_in: vf(P_in, res.X[0:-1])


def linear_vf(P, x): 

    return np.matmul(P, x.T).T


def plot_vf(P, vf): 

    plt.scatter(P[:,0], P[:,1], marker=".", color="red", s=200 )
   
    for i in range(np.size(P,0)):
        plt.annotate("P%d" % (i+1), (P[i,0], P[i,1]))

    # TODO pull 0 and 8 from min/max P 
    x,y = np.meshgrid(np.linspace(0, 8, 1000), np.linspace(0, 8, 1500))

    z = vf(np.stack((x,y), axis=2))

    z = z.T

    values_at_P = []
    for p in range(np.size(P,0)):
        values_at_P.append(vf(P[p,:]))

    values_at_P.sort()

    plt.contour(x,y,z, levels=values_at_P)

    plt.show()



def _build_ineq_constr_linear(P, vf):

    ineq_func = lambda x : _ineq_constr_linear(x, P, vf)

    return ineq_func

def _build_constr_linear(P, vf):

    ineq_constr_func = _build_ineq_constr_linear(P, vf);

    return lambda x : np.append(ineq_constr_func(x), _eq_constr_linear(x))


def _ineq_constr_linear(x, P, vf):
    if len(x.shape) == 1:
        return _ineq_constr_1D_linear(x, P, vf)
    else: 
        return _ineq_constr_2D_linear(x, P, vf)


def _ineq_constr_2D_linear(x, P, vf):

    ep = np.column_stack([x[:,-1]]) 
    pop_size = np.size(x,0)

    G = np.ones((pop_size, np.size(P,0)-1))*-99

    # Pair-wise compare each ranked member of P, seeing if our proposed utility 
    #  function increases monotonically as rank increases
    for p in range(np.size(P,0) - 1):

        current_P = vf(P[[p],:], x[:, 0:-1])
        next_P = vf(P[[p+1],:], x[:, 0:-1])

        G[:,[p]] = -(current_P - next_P) + ep

    return G


def _ineq_constr_1D_linear(x, P, vf):

    ep = x[-1]

    G = np.ones((1, np.size(P,0)-1))*-99

    # Pair-wise compare each ranked member of P, seeing if our proposed utility 
    #  function increases monotonically as rank increases
    for p in range(np.size(P,0) - 1):

        current_P = vf(P[[p],:], x[0:-1])
        next_P = vf(P[[p+1],:], x[0:-1])

        G[:,[p]] = -(current_P - next_P) + ep

    return G

def _obj_func(x): 

    # check if x is a 1D array or a 2D array
    if len(x.shape) == 1:
        ep = x[-1]
    else: 
        ep = np.column_stack([x[:,-1]])

    return -ep

def _sort_P(P, ranks): 
    P_with_rank = np.hstack((P, np.array([ranks]).T))

    P_sorted = P[P_with_rank[:, -1].argsort()]

    return P_sorted


# Constraint that states that the x values must add up to 1
def _eq_constr_linear(x): 

    if len(x.shape) == 1:
        eq_cons = sum(x[0:-1]) - 1
    else: 
        eq_cons = np.sum(x[:,0:-1],1, keepdims=True) - 1

    return eq_cons


class OptimizeVF(Problem): 

    def __init__(self, P, ranks, vf):
       
        # One var for each dimension of the object space, plus epsilon 
        n_var_vf = np.size(P, 1) + 1

        # it has one inequality constraints for every pair of solutions in P
        n_ieq_c_vf = np.size(P,0) - 1
       
        xl_vf = [0.0] * n_var_vf 
        xu_vf = [1.0] * n_var_vf
        
        # upper/lower bound on the epsilon variable is -1000/1000
        xl_vf[-1] = -1000
        xu_vf[-1] = 1000

        # TODO start everything at 0.5

        self.P = _sort_P(P, ranks)

        self.vf = vf

        super().__init__(n_var_vf, n_obj=1, n_ieq_constr=n_ieq_c_vf, n_eq_constr=1, xl=xl_vf, xu=xu_vf)

    def _evaluate(self, x, out, *args, **kwargs):

        ## Objective function: 
        obj = _obj_func(x)

        # The objective function above returns a negated version of epsilon 
        ep = -obj

        # maximize epsilon, or the minimum distance between each contour 
        out["F"] = obj

        ## Inequality
        # TODO for now, assuming there are no ties in the ranks

        ineq_func = _build_ineq_constr_linear(self.P, self.vf)

        out["G"] = ineq_func(x)
            
        ## Equality constraint that keeps sum of x under 1
        out["H"] = _eq_constr_linear(x)






