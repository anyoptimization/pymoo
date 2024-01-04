import pytest

from pymoo.util import value_functions as mvf
import numpy as np


## ----------------------- Constructor test ----------------------
# It should take in a set of PO points, and then return a function. 
# That function should take in a given PO point and return the 
# value of that point according to the decision maker
test_dummy_val_fnc_inputs = [
    (np.array([[1,2], [2,3]]), [1,2]),
    (np.array([[2,3], [3,2]]), [3,2])
]

@pytest.mark.parametrize('P, rankings', test_dummy_val_fnc_inputs)
def test_vf_constructor(P, rankings):

    val_fnc = mvf.create_linear_vf(P, rankings)

    assert val_fnc(P[0,:]) 
## ----------------------- Test Ranking -------------------------

## Tests whether the constructor is running correctly 
# Assumes that there is complete ordering (no ties in ranks)
test_test_prob_const_in_out = [
    (np.array([[1,2], [2,3], [4,5]]), [1,3,2], np.array([[1,2], [4,5], [2,3]])),
    (np.array([[2,3], [3,2], [7,8]]), [5,2,1], np.array([[7,8], [3,2], [2,3]]))
]

@pytest.mark.parametrize('P, rankings, output', test_test_prob_const_in_out)
def test_ranking(P, rankings, output):
    linear_vf = mvf.linear_vf

    vf_prob = mvf.OptimizeVF(P, rankings, linear_vf)

    ## Test whether the solutions are ranked by ranking 
    P_from_prob = vf_prob.P

    # Check the solutions 
    assert np.all(output == P_from_prob)
   

## TODO test the constructor with partial ordering 


## ------------------------ Objective function -----------------
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
    ),
    (
        np.array(
            [0.8, 0.1, -52]
        ), 
        52
    )
]


@pytest.mark.parametrize('x, obj', test_obj_in_out)
def test_obj(x, obj):

    result = mvf._obj_func(x)
    
    # Test whether or not the objective function simply negates the epsilon term of x (last element)
    assert np.all(obj == result)


## ------------- Test the inequality constraint for linear function ---------------
#  The expected values are pulled from the debugger of our linear.m file
test_linear_ineq_in_out = [

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


@pytest.mark.parametrize('x, P, ranks, expected_ineq_con', test_linear_ineq_in_out)
def test_ineq_constr_linear(x, P, ranks, expected_ineq_con):

    P_sorted = mvf._sort_P(P, ranks)

    result = mvf._ineq_constr_linear(x, P_sorted, mvf.linear_vf)

    # Test whether or not the constraint function matches our expected values   
    assert np.all(np.isclose(expected_ineq_con, result))


## ------------- Test the inequality constraints for polynomial VF -------------

test_poly_ineq_in_out = [

    (

        # Linear function values to optimize (x). This is one individual
        np.array([0.8, 0.22, 0.82, 0.94, 261, -351.91, 0.5]), 

        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9], 
                  [2.5, 4.1],    
                  [5.5, 2.5],      
                  [0.5, 5.2],     
                  [6.9, 1.8]]), 
        # Ranking of the P values, as per the decision maker 
        [1, 2, 3, 4, 5],
        # The constraint values, given the x
        np.array([[
            525.738,-697.202,     
            524.902,-697.916,
            526.95,-696.96,
            523.544,-698.522,
            527.916,-696.47,
            208.580844,-925.067768,1556.570032,-1970.154552]])
    ),
    (

        # Linear function values to optimize (x). This is one individual
        np.array([[0.8, 0.22, 0.82, 0.94, 261, -351.91, 0.5], 
                 [1, 0.34, 0.33, 0.49, -333.48, -360.15, 0.6]]), 

        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9], 
                  [2.5, 4.1],    
                  [5.5, 2.5],      
                  [0.5, 5.2],     
                  [6.9, 1.8]]), 
        # Ranking of the P values, as per the decision maker 
        [1, 2, 3, 4, 5],
        # The constraint values, given the x
        np.array([[
            525.738,-697.202,     
            524.902,-697.916,
            526.95,-696.96,
            523.544,-698.522,
            527.916,-696.47,
            208.580844,-925.067768,1556.570032,-1970.154552], 
            [
            -662.034, -717.201, 
            -663.066, -717.466, 
            -660.61, -717.26, 
            -664.692, -717.587, 
            -659.448, -717.141, 
            916.463922, -1897.582156, 3145.809604, -4056.540036
            ]])
    ),
]

@pytest.mark.parametrize('x, P, ranks, expected_ineq_con', test_poly_ineq_in_out)
def test_ineq_constr_poly(x, P, ranks, expected_ineq_con):

    sorted_P = mvf._sort_P(P, ranks) 

    constraints = mvf._ineq_constr_poly(x, P, mvf.poly_vf)
    
    assert np.all(np.isclose(expected_ineq_con, constraints))


## --------------- Test equality constraints -----------------------
test_eq_constr_poly_io = [
    (
        # Linear function values to optimize (x). This is one individual
        np.array([0.8, 0.2, 0.82, 0.94, 261, -351.91, 0.5]), 
        [0, -(0.82 + 0.94 - 1)]
    ),
    #(
    #    # Linear function values to optimize (x). This is one individual 
    #    np.array([[0.8, 0.22, 0.82, 0.94, 261, -351.91, 0.5], 
    #             [1, 0.34, 0.33, 0.49, -333.48, -360.15, 0.6]])

    #)
]


@pytest.mark.parametrize('x, expected_eq_con', test_eq_constr_poly_io)
def test_eq_constr_poly(x, expected_eq_con):

    constraints =  mvf._eq_constr_poly(x)

    assert np.all(np.isclose(expected_eq_con, constraints))




## --------------- Test calculating S from P and x --------------------------------
test_calc_S = [
    # K stuffed into an x matrix, one singleP 2-D P value (1D array)
    (
        np.array([0.8, 0.22, 0.82, 0.94, 261, -351.91]), 

        np.array([[3.6, 3.9]]), 

        np.array([525.738,-697.202])
        
    ),
    # K stuffed into an x matrix, a list of Ps (2D array)
    (
        np.array([0.8, 0.22, 0.82, 0.94, 261, -351.91]), 

        np.array([
            [3.6, 3.9], 
            [2.5, 4.1], 
            [5.5, 2.5]]), 

        np.array([  
                  [525.738,-697.202],
                  [524.902,-697.916],
                  [526.95, -696.96]
                  ])
        
    ),
    # K stuffed into an x matrix, a 2 dimensional space of Ps (3D array)
    (
        np.array([0.8, 0.22, 0.82, 0.94, 261, -351.91]), 
        np.array([
            [[3.6, 3.9], [2.5, 4.1]],
            [[5.5, 2.5], [0.5, 5.2]]
        ]),
        np.array([
            [[525.738, -697.202], [524.902, -697.916]],
            [[526.95, -696.96], [523.544, -698.522 ]]
        ])
    ),
    # K stuffed into an x matrix, a 2 dimensional space of Ps (3D array) with rectangular dimensions
    (
        np.array([0.8, 0.22, 0.82, 0.94, 261, -351.91]), 

        np.array([
            [[3.6, 3.9], [2.5, 4.1], [6.9, 1.8]],
            [[5.5, 2.5], [0.5, 5.2], [3.3, 2.1]]
        ]),
        np.array([
            [[525.738, -697.202], [524.902, -697.916], [527.916,-696.47]],
            [[526.95, -696.96], [523.544, -698.522 ], [525.102,-699.14]]
        ])
    ),
    # K stuffed into an x matrix, a 2 dimensional space of Ps (3D array) with rectangular dimensions
    (
        np.array([0.8, 0.22, 0.82, 0.94, 261, -351.91]), 

        np.array([
            [[3.6, 3.9], [2.5, 4.1]],
            [[5.5, 2.5], [0.5, 5.2]],
            [[6.9, 1.8], [3.3, 2.1]]
        ]),
        np.array([
            [[525.738, -697.202], [524.902, -697.916]],
            [[526.95, -696.96], [523.544, -698.522 ]],
            [[527.916,-696.47], [525.102,-699.14]]
        ])
    ) 

]

@pytest.mark.parametrize('x, P, expected_S', test_calc_S)
def test_calc_S(x, P, expected_S):


    assert np.shape(mvf._calc_S(P, x)) == np.shape(expected_S)

    S = mvf._calc_S(P, x)

    assert np.all(np.isclose(expected_S, S)) 


## --------------- Test the equality constraint for linear function --------------------
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
def test_eq_const_linear(x, P, ranks, expected_eq_constr):

    constr = mvf._eq_constr_linear(x)

    ## Test whether or not the constraint function matches our expected values   
    assert np.all(np.isclose(expected_eq_constr, constr))

## --------------- Test the linear value function ---------------

test_linear_vf_io = [
    # Test with a 1D P vector
    (
        np.array([0.5, 0.6]),
        np.array([
            [3.6, 3.9],
            [5.5, 2.5],
            [2.5, 4.1],
            [0.5, 5.2]
        ]),
        np.array([
          4.14,
          4.25,
          3.71,
          3.37
        ])
    ),
    # Test with a 2x2 2D P vector 
    (
        np.array([0.5, 0.6]),
        np.array([
            [[3.6, 3.9],[2.5, 4.1]],
            [[5.5, 2.5],[0.5, 5.2]]
        ]),
        np.array([
            [4.14, 3.71],
            [4.25, 3.37]
        ])
    ),
    # Test with a 3x2 2D P vector 
    (
        np.array([0.5, 0.6]),
        np.array([
            [[3.6, 3.9],[2.5, 4.1]],
            [[5.5, 2.5],[0.5, 5.2]],
            [[6.9, 1.8],[3.3, 2.1]],
        ]),
        np.array([
            [4.14, 3.71],
            [4.25, 3.37],
            [4.53, 2.91]
        ])
    ),
    (
        np.array([
            [0.5, 0.6],
            [0.4, 0.7]

            ]),
        np.array([
            [[3.6, 3.9],[2.5, 4.1]],
            [[5.5, 2.5],[0.5, 5.2]],
            [[6.9, 1.8],[3.3, 2.1]],
        ]),
        np.array([
            [[4.14, 4.17], [3.71, 3.87]],
            [[4.25, 3.95], [3.37, 3.84]],
            [[4.53, 4.02], [2.91, 2.79]]
        ])
    ),

]

@pytest.mark.parametrize('x, P, expected_value', test_linear_vf_io)
def test_linear_vf(x, P, expected_value):


    assert np.shape(expected_value) == np.shape(mvf.linear_vf(P, x))

    assert np.all(np.isclose(expected_value, mvf.linear_vf(P, x)))




## --------------- Test the polynomial value function ----------------

test_poly_vf_in_out = [

    (

        # Polynomial function values to optimize (x). These are five individuals
        np.array([
                [0.38,0.51,0.37,0.07,124.98,-284.6],
                [0.26,0.68,0.69,0.06,380.41,75.26],
                [0.87,0.06,0.73,0.03,134.89,-393.62],
                [0.53,0.46,0.85,0.52,-228,-377.7],
                [0.54,0.85,0.03,0.73,-296.88,200.68]
        ]), 


        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9]]), 

        # Expected value 
        np.array([-143781.4626, 117136.3531, -214281.6713, 339364.617, -237939.7818])
    ),
    (

        # Polynomial function values to optimize (x). These are five indentical individuals
        np.array(
                [0.38,0.51,0.37,0.07,124.98,-284.6],
        ), 


        # P, or the solutions to the problem we're trying to create a VF for 
        np.array([[3.6, 3.9]]), 

        # Expected value 
        np.array(-143781.4626)
    ),
    (
        np.array([0.8, 0.22, 0.82, 0.94, 261, -351.91]), 
        np.array([
            [[3.6, 3.9], [2.5, 4.1]],
            [[5.5, 2.5], [0.5, 5.2]]
        ]),
        np.array([
            [-366545.5851, -366337.5042],
            [-367263.072, -365707.002]
        ])
    ) ,
    # K stuffed into an x matrix, a 2 dimensional space of Ps (3D array) with rectangular dimensions
    (
        np.array([0.8, 0.22, 0.82, 0.94, 261, -351.91]), 

        np.array([
            [[3.6, 3.9], [2.5, 4.1], [6.9, 1.8]],
            [[5.5, 2.5], [0.5, 5.2], [3.3, 2.1]]
        ]),
        np.array([
            [-366545.5851, -366337.5042, -367677.6565],
            [-367263.072, -365707.002, -367119.8123]
        ])
    ),
    # K stuffed into an x matrix, a 2 dimensional space of Ps (3D array) with rectangular dimensions
    (
        np.array([0.8, 0.22, 0.82, 0.94, 261, -351.91]), 

        np.array([
            [[3.6, 3.9], [2.5, 4.1]],
            [[5.5, 2.5], [0.5, 5.2]],
            [[6.9, 1.8], [3.3, 2.1]]
        ]),
        np.array([
            [-366545.5851, -366337.5042],
            [-367263.072, -365707.002],
            [-367677.6565, -367119.8123]

        ])
    ) 

]

@pytest.mark.parametrize('x, P, expected_value', test_poly_vf_in_out)
def test_poly_vf(x, P, expected_value):
  
    assert np.shape(mvf.poly_vf(P, x)) == np.shape(expected_value)

    assert np.all(np.isclose(expected_value, mvf.poly_vf(P, x)))


## --------------- Smoke test for creating a VF with GA -----------------------
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
    vf = mvf.create_linear_vf(P, ranks, "ES")

## --------------- Smoke test for creating a VF with scipy ------------------


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
    vf = mvf.create_linear_vf(P, ranks, "scimin")





