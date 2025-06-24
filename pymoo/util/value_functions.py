import numpy as np
import pymoo
import scipy
from pymoo.optimize import minimize as moomin
from scipy.optimize import minimize as scimin
from scipy.optimize import OptimizeResult
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA
import math 
from operator import mul
from functools import reduce
from pymoo.termination.default import DefaultSingleObjectiveTermination
import sys


# Input 1: A list of non-dominated points 
# Input 2: The ranking of the given non-dominated points 
# Input 3: constraint function for optimizing the value func
# Input 4: the skeleton utility function that we're trying to optimize
def create_vf(P, ranks, ineq_constr, vf="linear", delta=0.1, eps_max=1000, method="trust-constr", verbose=False): 

    if vf == "linear":
        return create_linear_vf(P, ranks, delta, eps_max, method, verbose)
    elif vf == "poly":
        return create_poly_vf(P, ranks, delta, eps_max, method, verbose)

    else:
        raise ValueError("Value function '%d' not supported." % vf) 

    return lambda f_new:  np.sum(f_new)

def create_poly_vf(P, ranks, delta=0.1, eps_max=1000, method="trust-constr", verbose=False):

    if method == "trust-constr" or method == "SLSQP": 
        return create_vf_scipy_poly(P, ranks, delta, eps_max, method=method, verbose=verbose)
    elif method == "ES": 
        return create_vf_pymoo_poly(P, ranks, delta, eps_max, method=method, verbose=verbose)
    else: 
        raise ValueError("Optimization method %s not supported" % method) 



def create_linear_vf(P, ranks, delta=0.1, eps_max=1000, method="trust-constr"): 
    
    if method == "trust-constr" or method == "SLSQP": 
        return create_vf_scipy_linear(P, ranks, delta, eps_max, method)
    elif method == "ES": 
        return create_vf_pymoo_linear(P, ranks, delta, eps_max, method)
    else: 
        raise ValueError("Optimization method %s not supported" % method) 


def create_vf_scipy_poly(P, ranks, delta, eps_max, method="trust-constr", verbose=False):

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
    ranks.sort() 

    constr = NonlinearConstraint(_build_constr_poly(P_sorted, poly_vf, ranks, delta), ineq_lb, ineq_ub)
  
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
    x_ub.append(eps_max)
   
    bounds = Bounds(x_lb, x_ub)

    # Initial position
    x0 = [1] * (M**2 + M + 1)

    if method == 'trust-constr':
        # The trust-constr method always finds the decision space linear
        hess = lambda x: np.zeros((len(x0), len(x0)))
    else: 
        hess = None

    res = scimin(_obj_func, 
                 x0, 
                 constraints=constr, 
                 bounds=bounds, 
                 method=method, 
                 hess=hess)

    # package up results 
    vf =  lambda P_in: poly_vf(P_in, res.x[0:-1])
    params = res.x[0:-1]
    epsilon = res.x[-1]

    fit = _validate_vf(res, verbose)

    return vfResults(vf, params, epsilon, fit)


def create_vf_scipy_linear(P, ranks, delta, eps_max, method="trust-constr", verbose=False): 

    # Gathering basic info
    M = P.shape[1]
    
    # Sort P
    P_sorted = _sort_P(P, ranks)
    ranks.sort()

    # Inequality constraints
    lb = [-np.inf] * (P.shape[0] - 1)
    ub = [0] * (P.shape[0] - 1)

    # Equality constraints
    lb.append(0)
    ub.append(0)

    constr = NonlinearConstraint(_build_constr_linear(P_sorted, linear_vf, ranks, delta), lb, ub)

    # Bounds on x
    x_lb = []
    x_ub = []

    for m in range(M): 
        x_lb.append(0)
        x_ub.append(1)

    x_lb.append(-1000)
    x_ub.append(eps_max)
   
    bounds = Bounds(x_lb, x_ub)

    # Initial position
    x0 = [0.5] * (M+1)

    if method == 'trust-constr':
        # The trust-constr method always finds the decision space linear
        hess = lambda x: np.zeros((len(x0), len(x0)))
    else: 
        hess = None

    res = scimin(_obj_func, 
                 x0, 
                 constraints= constr,
                 bounds=bounds, 
                 method=method, 
                 hess=hess)

    # package up results
    vf =  lambda P_in: linear_vf(P_in, res.x[0:-1])
    params = res.x[0:-1]
    epsilon = res.x[-1]
    fit = _validate_vf(res, verbose)

    return vfResults(vf, params, epsilon, fit)

def create_vf_pymoo_linear(P, ranks, delta, eps_max, method="ES", verbose=False): 

    vf_prob = OptimizeLinearVF(P, ranks, delta, eps_max, linear_vf)

    if method == "ES":
        algorithm = ES()
    elif method == "GA": 
        algorithm = GA()
    else: 
        raise ValueError("Optimization method %s not supported" % method) 

    res = moomin(vf_prob,
        algorithm,
        ('n_gen', 200),
        verbose=verbose,
        seed=1)

    vf = lambda P_in: linear_vf(P_in, res.X[0:-1])

    if res.X is not None:
        params = res.X[0:-1]
        epsilon = res.X[-1]
    else: 
        params = None
        epsilon = -1000

    fit = _validate_vf(res, verbose)

    return vfResults(vf, params, epsilon, fit)


def create_vf_pymoo_poly(P, ranks, delta, eps_max, method="trust-constr", verbose=False):

    vf_prob = OptimizePolyVF(P, ranks, delta, eps_max, poly_vf)

    if method == "ES":
        algorithm = ES()
    elif method == "GA": 
        algorithm = GA()
    else: 
        raise ValueError("Optimization method %s not supported" % method) 

    res = moomin(vf_prob,
        algorithm,
        ('n_gen', 100),
        verbose=verbose,
        seed=1)

    vf = lambda P_in: poly_vf(P_in, res.X[0:-1])

    if res.X is not None:
        params = res.X[0:-1]
        epsilon = res.X[-1]
    else: 
        params = None
        epsilon = -1000

    fit = _validate_vf(res, verbose)

    return vfResults(vf, params, epsilon, fit)


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


def plot_vf(P, vf, show=True): 

    plt.scatter(P[:,0], P[:,1], marker=".", color="red", s=200 )
  

    for i in range(np.size(P,0)):
        plt.annotate("P%d" % (i+1), (P[i,0], P[i,1]))

    min_x = min(P[:,0])
    min_y = min(P[:,1])
    max_x = max(P[:,0])
    max_y = max(P[:,1])

    x,y = np.meshgrid(np.linspace(min_x, max_x, 1000), np.linspace(min_y, max_y, 1000))

    z = vf(np.stack((x,y), axis=2))

    values_at_P = []
    for p in range(np.size(P,0)):
        values_at_P.append(vf(P[p,:]))

    values_at_P.sort()

    plt.contour(x,y,z, levels=values_at_P)

    plt.colorbar()

    if show: 
        plt.show()

    return plt



## ---------------- Polynomial VF creation functions ------------------

def _ineq_constr_poly(x, P, vf, ranks, delta):
    if len(x.shape) == 1:
        return _ineq_constr_1D_poly(x, P, vf, ranks, delta)
    else: 
        return _ineq_constr_2D_poly(x, P, vf, ranks, delta)


def _build_ineq_constr_poly(P, vf, ranks, delta):

    ineq_func = lambda x : _ineq_constr_poly(x, P, vf, ranks, delta)

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


def _build_constr_poly(P, vf, ranks, delta): 

    ineq_constr_func = _build_ineq_constr_poly(P, vf, ranks, delta)

    return lambda x : np.append(ineq_constr_func(x), _eq_constr_poly(x))


def _ineq_constr_2D_poly(x, P, vf, ranks, delta):

    pop_size = np.size(x,0)

    P_count = P.shape[0]
    M = P.shape[1]

    S_constr_len = M * P_count 
    increasing_len = P_count - 1

    G = np.ones((pop_size, S_constr_len + increasing_len))*-99
   
    for xi in range(pop_size): 
    
        G[xi, :] = _ineq_constr_1D_poly(x[xi, :], P, vf, ranks, delta)

    return G


def _ineq_constr_1D_poly(x, P, vf, ranks, delta):

    ep = x[-1]

    P_count = P.shape[0]
    M = P.shape[1]

    S_constr_len = M * P_count 
    increasing_len = P_count - 1

    G = np.ones((1, S_constr_len + increasing_len))*-99

    # Checking to make sure each S term in the polynomial objective function is non-negative
    S = _calc_S(P, x[0:-1]) * -1

    G[:, 0:S_constr_len] = S.reshape(1, S_constr_len)


    # Pair-wise compare each ranked member of P, seeing if our proposed utility 
    #  function increases monotonically as rank increases
    for p in range(P_count - 1):

        current_P_val = vf(P[[p],:], x[0:-1])
        next_P_val = vf(P[[p+1],:], x[0:-1])

        current_rank = ranks[p]
        next_rank = ranks[p+1]

        if current_rank == next_rank: 
            # Handle ties
            G[:,[p + S_constr_len]] = np.abs(current_P_val - next_P_val) - delta*ep
        else: 
            G[:,[p + S_constr_len]] = -(current_P_val - next_P_val) + ep


    return G

## ---------------- Linear VF creation functions ------------------

def _build_ineq_constr_linear(P, vf, ranks, delta):

    ineq_func = lambda x : _ineq_constr_linear(x, P, vf, ranks, delta)

    return ineq_func

def _build_constr_linear(P, vf, ranks, delta):

    ineq_constr_func = _build_ineq_constr_linear(P, vf, ranks, delta);

    return lambda x : np.append(ineq_constr_func(x), _eq_constr_linear(x))


def _ineq_constr_linear(x, P, vf, ranks, delta):
    if len(x.shape) == 1:
        return _ineq_constr_1D_linear(x, P, vf, ranks, delta)
    else: 
        return _ineq_constr_2D_linear(x, P, vf, ranks, delta)


def _ineq_constr_2D_linear(x, P, vf, ranks, delta):

    ep = np.column_stack([x[:,-1]]) 
    pop_size = np.size(x,0)

    G = np.ones((pop_size, np.size(P,0)-1))*-99

    # Pair-wise compare each ranked member of P, seeing if our proposed utility 
    #  function increases monotonically as rank increases
    for p in range(np.size(P,0) - 1):


        current_P_val = vf(P[[p],:], x[:, 0:-1])
        next_P = vf(P[[p+1],:], x[:, 0:-1])

        current_rank = ranks[p]
        next_rank = ranks[p+1]

        
        if current_rank == next_rank: 
            # Handle ties 
            G[:,[p]] = np.abs(current_P_val.T - next_P.T) - delta*ep
        else: 
            # As vf returns, each column is an value of P for a given x in the population
            # We transpose to make each ROW the value of P
            G[:,[p]] = -(current_P_val.T - next_P.T) + ep


    return G


def _ineq_constr_1D_linear(x, P, vf, ranks, delta):

    ep = x[-1]

    G = np.ones((1, np.size(P,0)-1))*-99

    # Pair-wise compare each ranked member of P, seeing if our proposed utility 
    #  function increases monotonically as rank increases
    for p in range(np.size(P,0) - 1):

        current_P_val = vf(P[[p],:], x[0:-1])
        next_P = vf(P[[p+1],:], x[0:-1])

        current_rank = ranks[p]
        next_rank = ranks[p+1]

        if current_rank == next_rank: 
            # Handle ties 
            G[:,[p]] = np.abs(current_P_val - next_P) - delta*ep
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

    def __init__(self, P, ranks, delta, eps_max, vf):
       
        # One var for each dimension of the object space, plus epsilon 
        n_var_vf = np.size(P, 1) + 1

        # it has one inequality constraints for every pair of solutions in P
        n_ieq_c_vf = np.size(P,0) - 1
       
        xl_vf = [0.0] * n_var_vf 
        xu_vf = [1.0] * n_var_vf
        
        # upper/lower bound on the epsilon variable is -1000/1000
        xl_vf[-1] = -1000
        xu_vf[-1] = eps_max

        # TODO start everything at 0.5

        self.P = _sort_P(P, ranks)

        self.ranks = ranks
        self.ranks.sort()

        self.vf = vf

        self.ranks = ranks
        self.delta = delta

        super().__init__(n_var_vf, n_obj=1, n_ieq_constr=n_ieq_c_vf, n_eq_constr=1, xl=xl_vf, xu=xu_vf)

    def _evaluate(self, x, out, *args, **kwargs):

        ## Objective function: 
        obj = _obj_func(x)

        # The objective function above returns a negated version of epsilon 
        ep = -obj

        # maximize epsilon, or the minimum distance between each contour 
        out["F"] = obj

        ## Inequality

        ineq_func = _build_ineq_constr_linear(self.P, self.vf, self.ranks, self.delta)

        out["G"] = ineq_func(x)
            
        ## Equality constraint that keeps sum of x under 1
        out["H"] = _eq_constr_linear(x)

def _validate_vf(res, verbose):

    message = "" 

    if isinstance(res, pymoo.core.result.Result):
        success = np.all(res.G <= 0)
        epsilon = res.X[-1]
        if not success: 
            message = "Constraints not met\n"
        if epsilon < 0:
            message = message + "Epsilon negative\n"


    elif isinstance(res, OptimizeResult):
        success = res.success and res.constr_violation <= 0
        epsilon = res.x[-1]

        if not (res.constr_violation <= 0): 
            message = "Constraints not met."
        else:
            message = res.message
    else: 
        ValueError("Internal error: bad result objective given for validation")

    if epsilon < 0 or not success: 
    
        if verbose:
            sys.stderr.write("WARNING: Unable to fit value function\n")    
            sys.stderr.write(message + "\n")    

        return False
    else:
        return True


class OptimizePolyVF(Problem): 

    def __init__(self, P, ranks, delta, eps_max, vf):
      
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
        xu_vf[-1] = eps_max 

        # TODO start everything at 0.5

        self.P = _sort_P(P, ranks)

        self.ranks = ranks
        self.ranks.sort()

        self.vf = vf

        self.delta = delta

        super().__init__(n_var_vf, n_obj=1, n_ieq_constr=n_ieq_c_vf, n_eq_constr=M, xl=xl_vf, xu=xu_vf)

    def _evaluate(self, x, out, *args, **kwargs):

        ## Objective function: 
        obj = _obj_func(x)

        # The objective function above returns a negated version of epsilon 
        ep = -obj

        # maximize epsilon, or the minimum distance between each contour 
        out["F"] = obj

        ## Inequality
        ineq_func = _build_ineq_constr_poly(self.P, self.vf, self.ranks, self.delta)

        out["G"] = ineq_func(x)
            
        ## Equality constraint that keeps sum of x under 1
        out["H"] = _eq_constr_poly(x)


class vfResults(): 

    def __init__(self, vf, params, epsilon, fit): 

        self.vf = vf
        self.params = params
        self.epsilon = epsilon  
        self.fit = fit 



