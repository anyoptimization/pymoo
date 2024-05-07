import numpy as np
from pymoo.optimize import minimize as moomin
from scipy.optimize import minimize as scimin
from pymoo.core.problem import Problem
import pymoo.algorithms.soo.nonconvex.ga
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
from pymoo.algorithms.soo.nonconvex.es import ES
import math 
from operator import mul
from functools import reduce
from pymoo.termination.default import DefaultSingleObjectiveTermination

# Notes: 
# I'm using the suffix _vf to denote variables used in the value-function optimization 
# This is to avoid confuse it with variables being optimized in the main function 

# Input 1: A list of non-dominated points 
# Input 2: The ranking of the given non-dominated points 
# Input 3: constraint function for optimizing the value func
# Input 4: the skeleton utility function that we're trying to optimize
def create_vf(P, ranks, ineq_constr, vf="linear", algorithm="scimin", minimize=True): 

    if vf == "linear":
        return create_linear_vf(P, ranks, algorithm, minimize)
    else:
        raise ValueError("Value function '%d' not supported." % vf) 
    

    return lambda f_new:  np.sum(f_new)

def create_poly_vf(P, ranks, algorithm="scimin", minimize=True):
    
    if algorithm == "scimin": 
        return create_vf_scipy_poly(P, ranks, minimize)
    elif algorithm == "ES": 
        return create_vf_pymoo_poly(P, ranks, minimize)
    else: 
        raise ValueError("Algorithm %s not supported" % algorithm) 



def create_linear_vf(P, ranks, algorithm="scimin", minimize=True): 
    
    if algorithm == "scimin": 
        return create_vf_scipy_linear(P, ranks, minimize)
    elif algorithm == "ES": 
        return create_vf_pymoo_linear(P, ranks, minimize)
    else: 
        raise ValueError("Algorithm %s not supported" % algorithm) 


def create_vf_scipy_poly(P, ranks, minimize=True):

    # Gathering basic info
    M = P.shape[1]

    P_count = P.shape[0]

    # Inequality constraints - check that each term of S in our obj is non-negative for each P
    ineq_lb = [-np.inf] * (P_count*M)
    ineq_ub = [0] * (P_count*M)

    # Inequality constraints - check VF is monotonically increasing with user preference
    ineq_lb += [-np.inf] * (P_count - 1)
    ineq_ub += [0] * (P_count - 1)

    # Equality constraints (Make sure all terms in VF add up to 1 per term of product)
    for m in range(M): 
        ineq_lb.append(0)
        ineq_ub.append(0)

    P_sorted = _sort_P(P, ranks)

    constr = NonlinearConstraint(_build_constr_poly(P_sorted, poly_vf, minimize), ineq_lb, ineq_ub)
  
    # Bounds on x 
    x_lb = []
    x_ub = []
    for m in range(M**2):
        x_lb.append(0)
        x_ub.append(1)

    for m in range(M): 
        x_lb.append(-1000)
        x_ub.append(1000)

    x_lb.append(-1000)
    x_ub.append(1000)
   
    bounds = Bounds(x_lb, x_ub)

    x0 = [1] * (M**2 + M + 1)

    res = scimin(_obj_func, x0, constraints= constr, bounds = bounds)

    # package up results 
    vf =  lambda P_in: poly_vf(P_in, res.x[0:-1])
    params = res.x[0:-1]
    epsilon = res.x[-1]

    return vfResults(vf, params, epsilon)


def create_vf_scipy_linear(P, ranks, minimize=True): 

    # Inequality constraints
    lb = [-np.inf] * (P.shape[0] - 1)
    ub = [0] * (P.shape[0] - 1)

    # Equality constraints
    lb.append(0)
    ub.append(0)

    P_sorted = _sort_P(P, ranks)

    constr = NonlinearConstraint(_build_constr_linear(P_sorted, linear_vf, minimize), lb, ub)

    x0 = [0.5, 0.5, 0.5]

    res = scimin(_obj_func, x0, constraints= constr)

    # package up results
    vf =  lambda P_in: linear_vf(P_in, res.x[0:-1])
    params = res.x[0:-1]
    epsilon = res.x[-1]

    return vfResults(vf, params, epsilon)

def create_vf_pymoo_linear(P, ranks, minimize=True): 

    vf_prob = OptimizeLinearVF(P, ranks, linear_vf, minimize)

    algorithm = ES()

    res = moomin(vf_prob,
        algorithm,
        ('n_gen', 200),
        seed=1)

    vf = lambda P_in: linear_vf(P_in, res.X[0:-1])
    params = res.X[0:-1]
    epsilon = res.X[-1]

    return vfResults(vf, params, epsilon)


def create_vf_pymoo_poly(P, ranks, minimize=True):

    vf_prob = OptimizePolyVF(P, ranks, poly_vf, minimize)

    algorithm = ES()

    res = moomin(vf_prob,
        algorithm,
        ('n_gen', 100),
        seed=1)

    vf = lambda P_in: poly_vf(P_in, res.X[0:-1])
    params = res.X[0:-1]
    epsilon = res.X[-1]

    return vfResults(vf, params, epsilon)


def linear_vf(P, x): 

    return np.matmul(P, x.T)

def poly_vf(P, x): 

    # find out M
    M = P.shape[-1]

    result = []

    if len(x.shape) == 1:
        x_len = 1    
    else:
        x_len = x.shape[0]

    # Calculate value for each row of x 
    for xi in range(x_len):

        running_product = 1

        # Get current x 
        if x_len == 1:
            curr_x = x
        else: 
            curr_x = x[xi, :]

        S = _calc_S(P, curr_x)

        # Multiply all of the S terms together along the final axis 
        product = np.prod(S, axis = len(np.shape(S))-1)

        result.append(product)

    return np.squeeze(np.array(result))


def plot_vf(P, vf): 

    plt.scatter(P[:,0], P[:,1], marker=".", color="red", s=200 )
   
    for i in range(np.size(P,0)):
        plt.annotate("P%d" % (i+1), (P[i,0], P[i,1]))

    # TODO pull 0 and 8 from min/max P 
    x,y = np.meshgrid(np.linspace(0, 8, 1000), np.linspace(0, 8, 1500))

    z = vf(np.stack((x,y), axis=2))

    values_at_P = []
    for p in range(np.size(P,0)):
        values_at_P.append(vf(P[p,:]))

    values_at_P.sort()

    plt.contour(x,y,z, levels=values_at_P)

    plt.show()


## ---------------- Polynomial VF creation functions ------------------

def _ineq_constr_poly(x, P, vf, minimize=True):
    if len(x.shape) == 1:
        return _ineq_constr_1D_poly(x, P, vf, minimize)
    else: 
        return _ineq_constr_2D_poly(x, P, vf, minimize)


def _build_ineq_constr_poly(P, vf, minimize=True):

    ineq_func = lambda x : _ineq_constr_poly(x, P, vf, minimize)

    return ineq_func

def _eq_constr_poly(x):

    M = math.floor(math.sqrt(6))

    if len(x.shape) == 1:
        result = [] 
        for m in range(M): 
            result.append(-(sum(x[m*M:m*M+M]) - 1))

    else: 

        pop_size = np.size(x,0)

        result = []

        for xi in range(pop_size): 

            result_for_xi = [] 

            for m in range(M): 

                result_for_xi.append(-(sum(x[xi,m*M:m*M+M]) - 1))

            result.append(result_for_xi)

        result = np.array(result)

    return result


def _build_constr_poly(P, vf, minimize=True): 

    ineq_constr_func = _build_ineq_constr_poly(P, vf, minimize)

    return lambda x : np.append(ineq_constr_func(x), _eq_constr_poly(x))


def _ineq_constr_2D_poly(x, P, vf, minimize=True):

    pop_size = np.size(x,0)

    P_count = P.shape[0]
    M = P.shape[1]

    S_constr_len = M * P_count 
    increasing_len = P_count - 1

    G = np.ones((pop_size, S_constr_len + increasing_len))*-99
   
    for xi in range(pop_size): 
    
        G[xi, :] = _ineq_constr_1D_poly(x[xi, :], P, vf, minimize)

    return G


def _ineq_constr_1D_poly(x, P, vf, minimize=True):

    ep = x[-1]

    P_count = P.shape[0]
    M = P.shape[1]

    S_constr_len = M * P_count 
    increasing_len = P_count - 1

    G = np.ones((1, S_constr_len + increasing_len))*-99

    # Checking to make sure each S term in the polynomial objective function is non-negative
    current_constr = 0

    S = _calc_S(P, x[0:-1]) * -1

    G[:, 0:S_constr_len] = S.reshape(1, S_constr_len)


    # Pair-wise compare each ranked member of P, seeing if our proposed utility 
    #  function increases monotonically as rank increases
    for p in range(P_count - 1):

        current_P_val = vf(P[[p],:], x[0:-1])
        next_P_val = vf(P[[p+1],:], x[0:-1])

        if minimize: 
            G[:,[p + S_constr_len]] = -(next_P_val - current_P_val) + ep
        else: 
            G[:,[p + S_constr_len]] = -(current_P_val - next_P_val) + ep

    return G

## ---------------- Linear VF creation functions ------------------

def _build_ineq_constr_linear(P, vf, minimize=True):

    ineq_func = lambda x : _ineq_constr_linear(x, P, vf, minimize)

    return ineq_func

def _build_constr_linear(P, vf, minimize=True):

    ineq_constr_func = _build_ineq_constr_linear(P, vf, minimize);

    return lambda x : np.append(ineq_constr_func(x), _eq_constr_linear(x))


def _ineq_constr_linear(x, P, vf, minimize=True):
    if len(x.shape) == 1:
        return _ineq_constr_1D_linear(x, P, vf, minimize)
    else: 
        return _ineq_constr_2D_linear(x, P, vf, minimize)


def _ineq_constr_2D_linear(x, P, vf, minimize=True):

    ep = np.column_stack([x[:,-1]]) 
    pop_size = np.size(x,0)

    G = np.ones((pop_size, np.size(P,0)-1))*-99

    # Pair-wise compare each ranked member of P, seeing if our proposed utility 
    #  function increases monotonically as rank increases
    for p in range(np.size(P,0) - 1):


        current_P_val = vf(P[[p],:], x[:, 0:-1])
        next_P = vf(P[[p+1],:], x[:, 0:-1])

        # As vf returns, each column is an value of P for a given x in the population
        # We transpose to make each ROW the value of P
        if minimize: 
            G[:,[p]] = -(next_P.T - current_P_val.T) - ep
        else: 
            G[:,[p]] = -(current_P_val.T - next_P.T) + ep

    return G


def _ineq_constr_1D_linear(x, P, vf, minimize=True):

    ep = x[-1]

    G = np.ones((1, np.size(P,0)-1))*-99

    # Pair-wise compare each ranked member of P, seeing if our proposed utility 
    #  function increases monotonically as rank increases
    for p in range(np.size(P,0) - 1):

        current_P_val = vf(P[[p],:], x[0:-1])
        next_P = vf(P[[p+1],:], x[0:-1])

        if minimize: 
            G[:,[p]] = -(next_P - current_P_val) - ep
        else: 
            G[:,[p]] = -(current_P_val - next_P) + ep

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

# Can have several P instances, but assumes one x instance 
def _calc_S(P, x): 

    M = P.shape[-1]

    # reshape x into a matrix called k
    k = np.array(x[0:M*M].reshape(M,M), copy=True)
    k = k.T  
   
    # roll each column down to work with equation 5
    for col in range(1, k.shape[1]):
        k[:,[col]] = np.roll(k[:,[col]], col)


    l = x[M*M:(M*M)+M].reshape(M,1)
    l = l.T

    # Calc S for an arbitrary dimensional space of P 
    S = np.matmul(P, k) + (l  * 2)

    return np.squeeze(S)


# Constraint that states that the x values must add up to 1
def _eq_constr_linear(x): 

    if len(x.shape) == 1:
        eq_cons = sum(x[0:-1]) - 1
    else: 
        eq_cons = np.sum(x[:,0:-1],1, keepdims=True) - 1

    return eq_cons


# Makes a comparator for a given value function and the P that is ranked second in the 
def make_vf_comparator(vf, P_rank_2):

    return lambda P : vf_comparator(vf, P_rank_2, P)

def vf_comparator(vf, P_rank_2, P):

    reference_value = vf(P_rank_2)

    if reference_value > vf(P):
        return -1
    elif reference_value < vf(P): 
        return 1
    else: 
        return 0 


class OptimizeLinearVF(Problem): 

    def __init__(self, P, ranks, vf, minimize=True):
       
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

        self.minimize = minimize

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

        ineq_func = _build_ineq_constr_linear(self.P, self.vf, self.minimize)

        out["G"] = ineq_func(x)
            
        ## Equality constraint that keeps sum of x under 1
        out["H"] = _eq_constr_linear(x)




class OptimizePolyVF(Problem): 

    def __init__(self, P, ranks, vf, minimize=True):
      
        M = P.shape[1]

        P_count = P.shape[0]


        # One var for each dimension of the object space, plus epsilon 
        n_var_vf = (M**2 + M + 1)

        # it has one inequality constraints for every pair of solutions in P
        n_ieq_c_vf =  (P_count*M) + (P_count - 1)
       
        xl_vf = [0.0] * n_var_vf 
        xu_vf = [1.0] * n_var_vf
        
        # upper/lower bound on the epsilon variable is -1000/1000
        xl_vf[-1] = -1000
        xu_vf[-1] = 1000

        # TODO start everything at 0.5

        self.minimize = minimize 
        self.P = _sort_P(P, ranks)

        self.vf = vf

        super().__init__(n_var_vf, n_obj=1, n_ieq_constr=n_ieq_c_vf, n_eq_constr=M, xl=xl_vf, xu=xu_vf)

    def _evaluate(self, x, out, *args, **kwargs):

        ## Objective function: 
        obj = _obj_func(x)

        # The objective function above returns a negated version of epsilon 
        ep = -obj

        # maximize epsilon, or the minimum distance between each contour 
        out["F"] = obj

        ## Inequality
        # TODO for now, assuming there are no ties in the ranks

        ineq_func = _build_ineq_constr_poly(self.P, self.vf, self.minimize)

        out["G"] = ineq_func(x)
            
        ## Equality constraint that keeps sum of x under 1
        out["H"] = _eq_constr_poly(x)


class vfResults(): 

    def __init__(self, vf, params, epsilon): 

        self.vf = vf
        self.params = params
        self.epsilon = epsilon  
    

