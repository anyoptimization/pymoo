import pytest
from pymoo.algorithms.soo.nonconvex.es import ES
from scipy.optimize import minimize as scimin
from pymoo.optimize import minimize as moomin
from scipy.optimize import NonlinearConstraint
from pymoo.util import value_functions as vf
import numpy as np


## Global helper variables
dummy_inputs = (np.array([[2,3], [3,2], [7,8]]), [5,2,1])


## General test for function I/O. 
# It should take in a set of PO points, and then return a function. 
# That function should take in a given PO point and return the 
# value of that point according to the decision maker
test_dummy_val_fnc_inputs = [
    (np.array([[1,2], [2,3]]), [1,2], np.array([1,2]), 3),
    (np.array([[2,3], [3,2]]), [2,1], np.array([3,2]), 5)
]

@pytest.mark.parametrize('P, rankings, test_f, expected_value', test_dummy_val_fnc_inputs)
def test_dummy_vf(P, rankings, test_f, expected_value):

    val_fnc = vf.create_linear_vf(P, rankings)

    assert val_fnc(test_f) == expected_value



## ----------------------------------------------------

## Tests whether the constructor is running correctly 
# Assumes that there is complete ordering (no ties in ranks)
test_test_prob_const_in_out = [
    (np.array([[1,2], [2,3], [4,5]]), [1,3,2], np.array([[1,2], [4,5], [2,3]])),
    (np.array([[2,3], [3,2], [7,8]]), [5,2,1], np.array([[7,8], [3,2], [2,3]]))
]

@pytest.mark.parametrize('P, rankings, output', test_test_prob_const_in_out)
def test_prob_const(P, rankings, output):

    linear_vf = vf.linear_vf

    vf_prob = vf.OptimizeVF(P, rankings, linear_vf)

    ## Test whether the solutions are ranked by ranking 
    ranks_from_prob = vf_prob.P[:, -1]
    P_from_prob = vf_prob.P[:, 0:-1]

    rankings.sort()

    # Check the solutions 
    assert np.all(output == P_from_prob)

    # Check the ranks
    assert ranks_from_prob.tolist() == rankings
   

## TODO test the constructor with partial ordering 


## ----------------------------------------------------
## Test the objective function     
test_obj_in_out = [
    (
        np.array([
            [0.3, 0.1,  12], 
            [0.2, 0.5,  13], 
            [0.8, 0.1, -14]
        ]), 
        np.array([[-12, -13, 14]]).T
    ),
    (
        np.array([
            [0.3, 0.1,  73], 
            [0.2, 0.5,  22], 
            [0.8, 0.1, -52]
        ]), 
        np.array([[-73, -22, 52]]).T
    )
]


@pytest.mark.parametrize('x, obj', test_obj_in_out)
def test_obj(x, obj):

    linear_vf = vf.linear_vf

    vf_prob = vf.OptimizeVF(dummy_inputs[0], dummy_inputs[1], linear_vf)

    out = {}

    vf_prob._evaluate(x, out)
    
    # Test whether or not the objective function simply negates the epsilon term of x (last element)
    assert np.all(obj == out["F"])

## ----------------------------------------------------
## Test the internal objective function 
test_obj_func_in_out = [
    (
        np.array([
            [0.3, 0.1,  12], 
            [0.2, 0.5,  13], 
            [0.8, 0.1, -14]
        ]), 
        np.array([[-12, -13, 14]]).T
    ),
    (
        np.array(
            [0.8, 0.1, -52]
        ), 
        52
    )
]


@pytest.mark.parametrize('x, true_obj', test_obj_func_in_out)
def test_obj_func(x, true_obj):

    linear_vf = vf.linear_vf

    vf_prob = vf.OptimizeVF(dummy_inputs[0], dummy_inputs[1], linear_vf)

    obj = vf_prob._obj_func(x)

    # Test whether or not the objective function simply negates the epsilon term of x (last element)
    assert np.all(obj == true_obj)


## ------------- Test the inequality for linear function ---------------
#  The expected values are pulled from the debugger of our linear.m file
test_ineq_in_out = [

    (

        # Linear function values to optimize (x). This is two individuals
        np.array([
            [0.5,    0.5, 0.5], 
            [0.3780, 0.6220, 0.2072]
        ]), 

        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9], 
                  [2.5, 4.1],    
                  [5.5, 2.5],      
                  [0.5, 5.2],     
                  [6.9, 1.8]]), 
         

        # Ranking of the P values, as per the decision maker 
        [1, 2, 3, 4, 5],
        # The constraint values, given the x
        np.array([
            [0.05, 1.2, -0.65, 2.0], 
            [-0.0842, 0.346, -0.0034, 0.5116]
        ])
    ),
]

#    

@pytest.mark.parametrize('x, P, ranks, expected_ineq_con', test_ineq_in_out)
def test_ineq(x, P, ranks, expected_ineq_con):

    linear_vf = vf.linear_vf

    vf_prob = vf.OptimizeVF(P, ranks, linear_vf)

    out = {}

    vf_prob._evaluate(x, out)
    
    # Test whether or not the constraint function matches our expected values   
    assert np.all(np.isclose(expected_ineq_con, out["G"]))

## -------------------------------------------------------------------
test_build_ineq_constr_in_out = [

    (

        # Linear function values to optimize (x). This is two individuals
        np.array([
            [0.5,    0.5, 0.5], 
            [0.3780, 0.6220, 0.2072]
        ]), 

        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9], 
                  [2.5, 4.1],    
                  [5.5, 2.5],      
                  [0.5, 5.2],     
                  [6.9, 1.8]]), 
         

        # Ranking of the P values, as per the decision maker 
        [1, 2, 3, 4, 5],
        # The constraint values, given the x
        np.array([
            [0.05, 1.2, -0.65, 2.0], 
            [-0.0842, 0.346, -0.0034, 0.5116]
        ])
    ),
    (

        # Linear function values to optimize (x). This is two individuals
        np.array([0.3780, 0.6220, 0.2072]), 

        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9], 
                  [2.5, 4.1],    
                  [5.5, 2.5],      
                  [0.5, 5.2],     
                  [6.9, 1.8]]), 
         

        # Ranking of the P values, as per the decision maker 
        [1, 2, 3, 4, 5],
        # The constraint values, given the x
        np.array([-0.0842, 0.346, -0.0034, 0.5116])
    ),
]


@pytest.mark.parametrize('x, P, ranks, expected_ineq_con', test_build_ineq_constr_in_out)
def test_buildIneqFunc(x, P, ranks, expected_ineq_con):

    linear_vf = vf.linear_vf

    vf_prob = vf.OptimizeVF(P, ranks, linear_vf)

    out = {}

    ineqFunc = vf_prob._build_ineq_constr_linear()

    G = ineqFunc(x)
    
    # Test whether or not the constraint function matches our expected values   
    assert np.all(np.isclose(expected_ineq_con, G))


## -------------------------------------------------------------------
test_eq_constr_in_out = [

    (

        # Linear function values to optimize (x). This is two individuals in the population 
        np.array([
            [0.5,    0.5, 0.5], 
            [0.3780, 0.6220, 0.2072]
        ]), 

        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9], 
                  [2.5, 4.1],    
                  [5.5, 2.5],      
                  [0.5, 5.2],     
                  [6.9, 1.8]]), 
         

        # Ranking of the P values, as per the decision maker 
        [1, 2, 3, 4, 5],

        # The constraint values, given the x
        np.array([
            [0], 
            [0]
        ])
    ),
    (

        # Linear function values to optimize (x). This is two individuals in the population 
        np.array(
            [0.3780, 0.6220, 0.2072]
        ), 

        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9], 
                  [2.5, 4.1],    
                  [5.5, 2.5],      
                  [0.5, 5.2],     
                  [6.9, 1.8]]), 
         

        # Ranking of the P values, as per the decision maker 
        [1, 2, 3, 4, 5],

        # The constraint values, given the x
        0
    )
]


@pytest.mark.parametrize('x, P, ranks, expected_eq_constr', test_eq_constr_in_out)
def test_eqConst(x, P, ranks, expected_eq_constr):

    linear_vf = vf.linear_vf

    vf_prob = vf.OptimizeVF(P, ranks, linear_vf)

    constr = vf_prob._eq_constr_linear(x)

    ## Test whether or not the constraint function matches our expected values   
    assert np.all(np.isclose(expected_eq_constr, constr))

## ----------------------------------------------------
test_eq_const_in_out = [

    (

        # Linear function values to optimize (x). This is two individuals in the population 
        np.array([
            [0.5,    0.5, 0.5], 
            [0.3780, 0.6220, 0.2072]
        ]), 

        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9], 
                  [2.5, 4.1],    
                  [5.5, 2.5],      
                  [0.5, 5.2],     
                  [6.9, 1.8]]), 
         

        # Ranking of the P values, as per the decision maker 
        [1, 2, 3, 4, 5],

        # The constraint values, given the x
        np.array([
            [0], 
            [0]
        ])
    ),
]


@pytest.mark.parametrize('x, P, ranks, expected_eq_con', test_eq_const_in_out)
def test_eq_const(x, P, ranks, expected_eq_con):

    linear_vf = vf.linear_vf

    vf_prob = vf.OptimizeVF(P, ranks, linear_vf)

    out = {}

    vf_prob._evaluate(x, out)
    
    # Test whether or not the constraint function matches our expected values   
    assert np.all(np.isclose(expected_eq_con, out["H"]))

## ----------------------------------------------------
test_ga_in_out = [
    (


        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9], 
                  [2.5, 4.1],    
                  [5.5, 2.5],      
                  [0.5, 5.2],     
                  [6.9, 1.8]]), 
         

        # Ranking of the P values, as per the decision maker 
        [1, 2, 3, 4, 5]


    )
]

@pytest.mark.parametrize('P, ranks', test_ga_in_out)
def test_ga(P, ranks): 

    linear_vf = vf.linear_vf

    vf_prob = vf.OptimizeVF(P, ranks, linear_vf)

    algorithm = ES()
   
    res = moomin(vf_prob,
               algorithm,
               ('n_gen', 200),
               seed=1)

## ----------------------------------------------------


test_scipy_in_out = [
    (


        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9], 
                  [2.5, 4.1],    
                  [5.5, 2.5],      
                  [0.5, 5.2],     
                  [6.9, 1.8]]), 
         

        # Ranking of the P values, as per the decision maker 
        [1, 2, 3, 4, 5]


    )
]

@pytest.mark.parametrize('P, ranks', test_scipy_in_out)
def test_scipy(P, ranks): 

    linear_vf = vf.linear_vf

    vf_prob = vf.OptimizeVF(P, ranks, linear_vf)

    # Inequality constraints
    lb = [-np.inf] * (P.shape[0] - 1)
    ub = [0] * (P.shape[0] - 1)

    # Equality constraints
    lb.append(0)
    ub.append(0)

    constr = NonlinearConstraint(vf_prob._build_constr_linear(), lb, ub)

    x0 = [0.5, 0.5, 0.5]

    res = scimin(vf_prob._obj_func, x0, constraints= constr)





