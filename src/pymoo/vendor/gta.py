"""Dynamic Multi-objective Optimization Problems
This Library is a GTA(a and m) benchmarks from the website of the author,
and the implementation of the FDA, DIMP, DMOP, HE benchmarks of the CEC2015 competition
nt: severity of change
taut: frequency of change
tauT: maximum number of generation
tau : current generation

"""
# !/bin/python

import numpy as np
from random import randint
from math import floor, fabs, sin, pi, cos, sqrt

## Parameter configuration ##
LOWER_BOUND = [0.0] + 20 * [-1.0]
UPPER_BOUND = 21 * [1.0]
ERR_MSG = "x is outside decision boundary or dimension of x is not correct"
DELTA_STATE = 1


## Define component functions ##
def beta_uni(x, t, g, obj_num=2):
    """This function is used to calculate the unimodal beta function. Input are
    the decision variable (x), time (t) and g function (g).
    """
    beta = [0.0] * obj_num
    for i in range(obj_num - 1, len(x)):
        beta[(i + 1) % obj_num] += (x[i] - g(x, t)) * (x[i] - g(x, t))

    beta = [(2.0 / int(len(LOWER_BOUND) / obj_num)) * b for b in beta]
    return beta


def beta_multi(x, t, g, obj_num=2):
    """This function is used to calculate the multi-modal beta function. Input
    are the decision variable (x), time (t) and g function (g).
    """
    beta = [0.0] * obj_num
    for i in range(obj_num - 1, len(x)):
        beta[(i + 1) % obj_num] += (x[i] - g(x, t)) * (x[i] - g(x, t)) * \
                                   (1 + np.abs(np.sin(4 * np.pi * (x[i] - g(x, t)))))

    beta = [(2.0 / int(len(LOWER_BOUND) / obj_num)) * b for b in beta]
    return beta


def beta_mix(x, t, g, obj_num=2):
    """This function is used to calculate the mixed unimodal and multi-modal
    beta function. Input are the decision variable (x), time (t) and g function
    (g).
    """
    beta = [0.0] * obj_num
    k = int(abs(5.0 * (int(DELTA_STATE * int(t) / 5.0) % 2) - (DELTA_STATE * int(t) % 5)))

    for i in range(obj_num - 1, len(x)):
        temp = 1.0 + (x[i] - g(x, t)) * (x[i] - g(x, t)) - np.cos(2 * np.pi * k * (x[i] - g(x, t)))
        beta[(i + 1) % obj_num] += temp
    beta = [(2.0 / int(len(LOWER_BOUND) / obj_num)) * b for b in beta]
    return beta


def alpha_conv(x):
    """This function is used to calculate the alpha function with convex POF.
    Input is decision variable (x).
    """
    return [x[0], 1 - np.sqrt(x[0])]


def alpha_disc(x):
    """This function is used to calculate the alpha function with discrete POF.
    Input is decision variable (x).
    """
    return [x[0], 1.5 - np.sqrt(x[0]) - 0.5 * np.sin(10 * np.pi * x[0])]


def alpha_mix(x, t):
    """This function is used to calculate the alpha function with mixed
    continuous POF and discrete POF.
    """
    k = int(abs(5.0 * (int(DELTA_STATE * int(t) / 5.0) % 2) - (DELTA_STATE * int(t) % 5)))
    return [x[0], 1 - np.sqrt(x[0]) + 0.1 * k * (1 + np.sin(10 * np.pi * x[0]))]


def alpha_conf(x, t):
    """This function is used to calculate the alpha function with time-varying
    conflicting objective. Input are decision variable (x) and time (t).
    """
    k = int(abs(5.0 * (int(DELTA_STATE * int(t) / 5.0) % 2) - (DELTA_STATE * int(t) % 5)))
    return [x[0], 1 - np.power(x[0], \
                               np.log(1 - 0.1 * k) / np.log(0.1 * k + np.finfo(float).eps))]


def alpha_conf_3obj_type1(x, t):
    """This function is used to calculate the alpha unction with time-varying
    conflicting objective (3-objective, type 1). Input are decision variable
    (x) and time (t).
    """
    k = int(abs(5.0 * (int(DELTA_STATE * int(t) / 5.0) % 2) - (DELTA_STATE * int(t) % 5)))
    alpha1 = fix_numerical_instability(np.cos(0.5 * x[0] * np.pi) * np.cos(0.5 * x[1] * np.pi))
    alpha2 = fix_numerical_instability(np.cos(0.5 * x[0] * np.pi) * np.sin(0.5 * x[1] * np.pi))
    alpha3 = fix_numerical_instability(np.sin(0.5 * x[0] * np.pi + 0.25 * (k / 5.0) * np.pi))
    return [alpha1, alpha2, alpha3]


def alpha_conf_3obj_type2(x, t):
    """This function is used to calculate the alpha unction with time-varying
    conflicting objective (3-objective, type 2). Input are decision variable (x)
    and time (t).
    """
    k = int(abs(5.0 * (int(DELTA_STATE * int(t) / 5.0) % 2) - (DELTA_STATE * int(t) % 5)))
    k_ratio = (5.0 - k) / 5.0
    alpha1 = fix_numerical_instability(np.cos(0.5 * x[0] * np.pi) * np.cos(0.5 * x[1] * np.pi * k_ratio))
    alpha2 = fix_numerical_instability(np.cos(0.5 * x[0] * np.pi) * np.sin(0.5 * x[1] * np.pi * k_ratio))
    alpha3 = fix_numerical_instability(np.sin(0.5 * x[0] * np.pi))
    return [alpha1, alpha2, alpha3]


def g(x, t):
    """This function is used to calculate the g function used in the paper.
    Input are decision variable (x) and time (t).
    """
    return np.sin(0.5 * np.pi * (t - x[0]))


## Utility functions ##
def check_boundary(x, upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND):
    """Check the dimension of x and whether it is in the decision boundary. x is
    decision variable, upper_bound and lower_bound are upperbound and lowerbound
    lists of the decision space
    """
    if len(x) != len(upper_bound) or len(x) != len(lower_bound):
        return False

    output = True
    for e, upp, low in zip(x, upper_bound, lower_bound):
        output = output and (e >= low) and (e <= upp)
    return output


def check_boundary_3obj(x, upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND):
    """Check the dimension of x and whether it is in the decision boundary. x is
    decision variable, upper_bound and lower_bound are upperbound and lowerbound
    lists of the decision space
    """
    lower_bound = [0.0] + lower_bound
    upper_bound = [1.0] + upper_bound
    if len(x) != len(upper_bound) or len(x) != len(lower_bound):
        return False

    output = True
    for e, upp, low in zip(x, upper_bound, lower_bound):
        output = output and (e >= low) and (e <= upp)
    return output


def fix_numerical_instability(x):
    """Check whether x is close to zero, sqrt(0.5) or not. If it is close to
    these two values, changes x to the value. Otherwise, return x.
    """
    if np.allclose(0.0, x):
        return 0.0

    if np.allclose(np.sqrt(0.5), x):
        return np.sqrt(0.5)
    return x


def additive(alpha, beta):
    """Additive form of the benchmark problem.
    """
    return [a + b for a, b in zip(alpha, beta)]


#    return [alpha[0] + beta[0], alpha[1] + beta[1]]


def multiplicative(alpha, beta):
    """Multiplicative form of the benchmark problem.
    """
    return [a * (1 + b) for a, b in zip(alpha, beta)]


#    return [alpha[0]*(1 + beta[0]), alpha[1]*(1 + beta[1])]


## Benchmark functions ##
def DB1a(x, t):
    """DB1a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_uni(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB1m(x, t):
    """DB1m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_uni(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB2a(x, t):
    """DB2a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_multi(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB2m(x, t):
    """DB2m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_multi(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB3a(x, t):
    """DB3a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_mix(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB3m(x, t):
    """DB3m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_mix(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB4a(x, t):
    """DB4a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_disc(x)
        beta = beta_mix(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB4m(x, t):
    """DB4m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_disc(x)
        beta = beta_mix(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB5a(x, t):
    """DB5a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_mix(x, t)
        beta = beta_multi(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB5m(x, t):
    """DB5m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_mix(x, t)
        beta = beta_multi(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB6a(x, t):
    """DB6a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_mix(x, t)
        beta = beta_mix(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB6m(x, t):
    """DB6m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_mix(x, t)
        beta = beta_mix(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB7a(x, t):
    """DB7a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf(x, t)
        beta = beta_multi(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB7m(x, t):
    """DB7m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf(x, t)
        beta = beta_multi(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB8a(x, t):
    """DB8a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf(x, t)
        beta = beta_mix(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB8m(x, t):
    """DB8m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf(x, t)
        beta = beta_mix(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB9a(x, t):
    """DB9a dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj_type1(x, t)
        beta = beta_multi(x, t, g, obj_num=3)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB9m(x, t):
    """DB9m dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj_type1(x, t)
        beta = beta_multi(x, t, g, obj_num=3)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB10a(x, t):
    """DB10a dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj_type1(x, t)
        beta = beta_mix(x, t, g, obj_num=3)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB10m(x, t):
    """DB10m dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj_type1(x, t)
        beta = beta_mix(x, t, g, obj_num=3)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB11a(x, t):
    """DB11a dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj_type2(x, t)
        beta = beta_multi(x, t, g, obj_num=3)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB11m(x, t):
    """DB11m dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj_type2(x, t)
        beta = beta_multi(x, t, g, obj_num=3)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB12a(x, t):
    """DB12a dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj_type2(x, t)
        beta = beta_mix(x, t, g, obj_num=3)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB12m(x, t):
    """DB12m dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj_type2(x, t)
        beta = beta_mix(x, t, g, obj_num=3)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def fda2_deb(x, t):
    f1 = x[0]
    H = 2 * np.sin(0.5 * np.pi * (t - 1))
    XII = x[1:6]
    XIII = x[6:13]
    g = 1 + np.sum(np.power(XII, 2))
    Htemp = np.sum(np.power((XIII - H / 4), 2))
    h = 1 - np.power((f1 / g), np.power(2, H + Htemp))
    f2 = g * h
    return [f1, f2]


def FDA4(X, t):
    """FDA4 dynamic benchmark problem
    """
    XII = X[2:]
    G = fabs(sin(0.5 * pi * t))
    g = sum([pow(xi - G, 2) for xi in XII])
    f1 = (1 + g) * cos(X[0] * pi / 2) * cos(X[1] * pi / 2)
    f2 = (1 + g) * cos(X[0] * pi / 2) * sin(X[1] * pi / 2)
    f3 = (1 + g) * sin(X[0] * pi / 2)
    return [f1, f2, f3]


def FDA5(X, t):
    """FDA5 dynamic benchmark problem
    """
    XII = X[2:]
    G = fabs(sin(0.5 * pi * t))
    g = G + sum([pow(xi - G, 2) for xi in XII])
    F = 1 + 100 * pow(sin(0.5 * pi * t), 4)
    y = lambda i: pow(X[i], F)
    f1 = (1 + g) * cos(y(0) * pi / 2) * cos(y(1) * pi / 2)
    f2 = (1 + g) * cos(y(0) * pi / 2) * sin(y(1) * pi / 2)
    f3 = (1 + g) * sin(y(0) * pi / 2)
    return [f1, f2, f3]


def DIMP2(X, t):
    """DIMP2 dynamic benchmark problem
    """
    n = len(X)
    XII = X[1:]
    g = 1.0 + 2.0 * (len(XII))
    for k in range(1, n):
        G = sin(pow(0.5 * pi * t + 2.0 * pi * float(k + 1) / float(n + 1.0), 2))
        g += pow(X[k] - G, 2) - 2.0 * cos(3.0 * pi * (X[k] - G))
    f1 = X[0]
    h = 1 - sqrt(f1 / g)
    f2 = g * h
    return [f1, f2]


def dMOP2(X, t):
    """dMOP2 dynamic benchmark problem
    """
    XII = X[1:]
    G = sin(0.5 * pi * t)
    g = 1 + 9 * sum([pow(xi - G, 2) for xi in XII])
    H = 0.75 * sin(0.5 * pi * t) + 1.25
    f1 = X[0]
    h = 1 - pow((f1 / g), H)
    f2 = g * h
    return [f1, f2]


def dMOP3(X, tau, nt, taut, r, rIteration):
    """dMOP3 dynamic benchmark problem
    """
    if tau % taut == 0 and rIteration != tau:
        r = randint(0, 9)
        rIteration = tau

    XII = X[:r] + X[r + 1:]
    t = float(1) / float(nt)
    t = t * floor(float(tau) / float(taut))
    G = sin(0.5 * pi * t)
    g = 1 + 9 * sum([pow(xi - G, 2) for xi in XII])
    H = 0.75 * sin(0.5 * pi * t)
    f1 = X[r]
    f2 = 1 - pow((f1 / g), H)
    return [f1, f2, r, rIteration]


def HE2(X, t):
    """HE2 dynamic benchmark problem
    """
    n = 30
    XII = X[1:]
    H = 0.75 * sin(0.5 * pi * t) + 1.25
    g = 1 + (9 / (n - 1)) * sum(XII)
    f1 = X[0]
    h = 1 - pow(sqrt(f1 / g), H) - pow(f1 / g, H) * sin(10 * pi * f1)
    f2 = g * h
    return [f1, f2]


def HE7(X, t):
    """HE7 dynamic benchmark problem
    """

    def _f1(input1):
        value = input1[0]
        ssum = 0.0
        index = 0
        for k in range(2, len(input1), 2):
            val = 6 * pi * value + k * pi / len(input1)
            ssum += pow(input1[k] - (0.3 * value * value * cos(4 * val) + 0.6 * value) * cos(val), 2)
            index += 1
        ssum *= 2.0 / index
        ssum += value
        return ssum

    def _g(input1):
        value = input1[0]
        ssum = 0.0
        index = 0
        for k in range(1, len(input1), 2):
            val = 6 * pi * value + k * pi / len(input1)
            ssum += pow(input1[k] - (0.3 * value * value * cos(4 * val) + 0.6 * value) * sin(val), 2)
            index += 1
        ssum *= 2.0 / index
        ssum += 2.0 - sqrt(value)
        return ssum

    f1 = _f1(X)
    g = _g(X)
    H = 0.75 * sin(0.5 * pi * t) + 1.25
    h = 1 - pow(f1 / g, H)
    f2 = g * h
    return [f1, f2]


def HE9(X, t):
    """HE9 dynamic benchmark problem
    """

    def _f1(input1):
        value = input1[0]
        ssum = 0.0
        index = 0
        for k in range(2, len(input1), 2):
            ssum += pow(input1[k] - sin(6 * pi * value + k * pi / len(input1)), 2)
            index += 1
        ssum *= 2.0 / index
        ssum += value
        return ssum

    def _g(input1):
        value = input1[0]
        ssum = 0.0
        index = 0
        for k in range(1, len(input1), 2):
            ssum += pow(input1[k] - sin(6 * pi * value + k * pi / len(input1)), 2)
            index += 1
        ssum *= 2.0 / index
        ssum += 2.0 - pow(value, 2)
        return ssum

    f1 = _f1(X)
    g = _g(X)
    H = 0.75 * sin(0.5 * pi * t) + 1.25
    h = 1 - pow(f1 / g, H)
    f2 = g * h
    return [f1, f2]
